"""M4：換線成本 × 庫存的取捨線（完全資訊）、實際計畫定位、滾動計畫。

    python scripts/run_m4.py      # 需先跑過 run_m0_m1.py 與 run_m2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402
import kpi  # noqa: E402
import schedule as M  # noqa: E402
import simulate as S  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_m3  # noqa: E402

RESULTS = ROOT / "results"
SETUP_COSTS = (0, 1, 2, 5, 10, 20, 50, 100, 200)
ROLLING_SETUP_COST = 20          # 事前選定，不看結果挑
SS_ALPHA, SS_LEAD = 0.95, 2      # 滾動計畫的安全庫存沿用 M3 基準（empirical、R=7、L=2）
START, END = "2023-05-21", "2023-08-05"   # 與 M3 量測期相同的 11 週（週日到週六）
HOLD, PENALTY = 1.0, 10.0


def _r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main() -> None:
    flows, skus, _ = data.load()
    meta = skus.set_index("sku")
    bt = pd.read_csv(RESULTS / "m2_backtest.csv", parse_dates=["origin", "date"])
    prod_all = data.pivot(flows, "production", "qty_ton")
    # 整段期間從未生產的 SKU 不在這些產線上（可能外購或他廠供應），排除在排程之外
    never_made = sorted(s for s in bt.sku.unique() if prod_all[s].sum() == 0)
    active = sorted(s for s in bt.sku.unique() if s not in never_made)
    tpu = meta.ton_per_unit.reindex(active)
    groups = meta.group.reindex(active).tolist()

    prod_d = data.pivot(flows, "production", "qty_ton")[active]
    order_d = data.pivot(flows, "order", "qty_ton")[active]
    weeks = lambda df: df.loc[START:END].resample("W-SAT").sum()
    prod_w, order_w = weeks(prod_d), weeks(order_d)
    assert len(prod_w) == 11, len(prod_w)

    # 期初淨庫存（下限）：讓全期「期初 + 累計生產 − 累計訂單」不為負的最小值，再推到量測起點前一天
    open_lb = np.array([kpi.opening_lower_bound(prod_d[s], order_d[s]) for s in active])
    pre = prod_d.loc[:pd.Timestamp(START) - pd.Timedelta(days=1)]
    pre_o = order_d.loc[:pd.Timestamp(START) - pd.Timedelta(days=1)]
    net0 = open_lb + (pre.sum() - pre_o.sum()).to_numpy()

    weekly_prod_all = prod_d.resample("W-SAT").sum()
    weekly_prod_all = weekly_prod_all.iloc[1:-1]                     # 去掉頭尾殘週
    cap = {g: float(weekly_prod_all.T.groupby(groups).sum().T[g].quantile(0.95)) for g in set(groups)}
    min_batch = np.array([prod_d[s][prod_d[s] > 0].median() for s in active])

    D = order_w.T.to_numpy()
    P_actual = prod_w.T.to_numpy()
    T = D.shape[1]
    actual = M.evaluate(P_actual, D, net0)
    terminal = actual["end_net"]

    frontier = []
    for s in SETUP_COSTS:
        plan = M.solve(D, groups, cap, min_batch, net0, s, hold=HOLD, penalty=PENALTY, terminal=terminal)
        ev = M.evaluate(np.where(plan.y > 0.5, plan.x, 0), D, net0)
        frontier.append({"setup_cost": s, "status": plan.status, "mip_gap": _r(plan.mip_gap, 4),
                         "objective": _r(plan.objective), "setups": ev["setups"],
                         "mean_inventory_t": _r(ev["mean_inventory_t"]),
                         "mean_backlog_t": _r(ev["mean_backlog_t"]),
                         "actual_plan_cost_same_s": _r(M.cost(actual, s, HOLD, PENALTY, T))})
    frontier = pd.DataFrame(frontier)
    frontier.to_csv(RESULTS / "m4_frontier.csv", index=False)

    # 實際計畫落在哪：同樣換線次數下，取捨線上的庫存是多少
    f = frontier.sort_values("setups")
    inv_at_actual_setups = float(np.interp(actual["setups"], f.setups, f.mean_inventory_t))

    # 滾動計畫：星期平均的 7 天預測加總成週預測（起點＝每週的前一個週六）
    origins = pd.date_range(pd.Timestamp(START) - pd.Timedelta(days=1), periods=T, freq="7D")
    fc = (bt[bt.origin.isin(origins)].groupby(["origin", "sku"]).weekday_mean.sum()
          .unstack().reindex(index=origins, columns=active))
    F_w = (fc * tpu).T.to_numpy()
    # 安全庫存（噸）：每個起點只用該起點以前的誤差（同 M3）
    _, _, all_origins, scale, resid, cls, _ = run_m3.build_inputs()
    ss_w = np.zeros((len(active), T))
    for k, o in enumerate(origins):
        for i, sku in enumerate(active):
            pool = S.past_pool(resid["weekday_mean"], o, cls.index[cls == cls[sku]])
            q = S.safety_factor(pool, SS_ALPHA, "empirical")
            ss_w[i, k] = q * scale.loc[o, sku] * np.sqrt((7 + SS_LEAD) / 7) * tpu[sku]

    rolls = {}
    for name, ss in (("no_safety_stock", None), ("with_safety_stock", ss_w)):
        rp, rs = M.rolling(F_w, D, groups, cap, min_batch, net0, ROLLING_SETUP_COST,
                           safety=ss, hold=HOLD, penalty=PENALTY)
        ev = M.evaluate(rp, D, net0)
        rolls[name] = {"statuses": sorted(set(rs)), "setups": ev["setups"],
                       "mean_inventory_t": _r(ev["mean_inventory_t"]),
                       "mean_backlog_t": _r(ev["mean_backlog_t"]),
                       "cost": _r(M.cost(ev, ROLLING_SETUP_COST, HOLD, PENALTY, T))}
    perfect = frontier.set_index("setup_cost").loc[ROLLING_SETUP_COST]

    out = {
        "design": {"weeks": T, "start": START, "end": END, "skus": len(active),
                   "excluded_never_produced": never_made,
                   "lines": {g: int(groups.count(g)) for g in sorted(set(groups))},
                   "capacity_t_per_week": {g: _r(v, 1) for g, v in sorted(cap.items())},
                   "min_batch_t": {"median": _r(np.median(min_batch)), "max": _r(min_batch.max())},
                   "hold": HOLD, "penalty": PENALTY, "setup_costs": SETUP_COSTS,
                   "net0_total_t_lower_bound": _r(net0.sum(), 1),
                   "solver": "HiGHS via scipy.optimize.milp"},
        "actual_plan": {"setups": actual["setups"], "mean_inventory_t": _r(actual["mean_inventory_t"]),
                        "mean_backlog_t": _r(actual["mean_backlog_t"])},
        "frontier": frontier.to_dict(orient="records"),
        "frontier_inventory_at_actual_setups_t": _r(inv_at_actual_setups),
        "all_optimal": bool((frontier.status == "optimal").all()),
        "rolling": {"setup_cost": ROLLING_SETUP_COST, "forecast": "weekday_mean", "horizon_weeks": 4,
                    "safety_stock": {"alpha": SS_ALPHA, "kind": "empirical", "lead": SS_LEAD,
                                     "mean_total_t": _r(ss_w.sum(axis=0).mean(), 1)},
                    **rolls},
        "perfect_info_same_setup_cost": {"setups": int(perfect.setups),
                                         "mean_inventory_t": perfect.mean_inventory_t,
                                         "mean_backlog_t": perfect.mean_backlog_t,
                                         "cost": perfect.objective},
        "actual_cost_same_setup_cost": _r(M.cost(actual, ROLLING_SETUP_COST, HOLD, PENALTY, T)),
        "weekly_order_t_mean": _r(D.sum(axis=0).mean(), 1),
    }
    (RESULTS / "m4_schedule.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(frontier.to_string(index=False))
    print(json.dumps({k: out[k] for k in ("design", "actual_plan", "frontier_inventory_at_actual_setups_t",
                                          "rolling", "perfect_info_same_setup_cost",
                                          "actual_cost_same_setup_cost", "weekly_order_t_mean")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
