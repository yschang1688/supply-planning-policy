"""M3：庫存政策實驗格點、H1／H2 判定、Monte Carlo 前置期延誤。

    python scripts/run_m3.py      # 需先跑過 run_m0_m1.py 與 run_m2.py
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
import forecast as F  # noqa: E402
import simulate as S  # noqa: E402

RESULTS = ROOT / "results"
ALPHAS = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99)
KINDS = ("normal_classic", "normal", "empirical")
POLICIES = {"RS": 7, "sS": 1}           # 政策 → 安全庫存的保護期中「檢視週期」那一段
LEADS = (1, 2, 3, 5, 7)
BASE = dict(method="weekday_mean", policy="RS", kind="empirical", lead=2)
LEAD_SWEEP_METHODS = ("weekday_mean", "lgbm_calendar")
WARMUP = 4                               # 前 4 個起點只用來累積誤差
REASONABLE = [m for m in F.METHODS if m not in ("ma28", "seasonal_naive")]
N_BOOT, N_MC = 1000, 200
FILL_TARGET = 0.98     # 等服務比較點：每個方法的目標曲線都跨過 0.98（0.95 時多數方法在 α=0.80 就已超過）


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def build_inputs():
    flows, skus, _ = data.load()
    meta = skus.set_index("sku")
    table = pd.read_csv(RESULTS / "m1_sku_table.csv", index_col="sku")
    bt = pd.read_csv(RESULTS / "m2_backtest.csv", parse_dates=["origin", "date"])
    active = sorted(bt.sku.unique())
    order = data.pivot(flows, "order")[active]
    origins = sorted(bt.origin.unique())

    # 起點當天已知的尺度：前 28 天平均日需求 × 7
    scale = {o: order.loc[:o].tail(28).mean() * 7 for o in origins}
    scale = pd.DataFrame(scale).T                                     # origin × sku
    wk = bt.groupby(["origin", "sku"])[["actual", *F.METHODS]].sum()
    resid = {m: ((wk.actual - wk[m]).unstack() / scale).reindex(columns=active)
             for m in F.METHODS}                                        # origin × sku
    cls = table.xyz.reindex(active).map(lambda x: "X" if x == "X" else "YZ")
    return order, bt, origins, scale, resid, cls, meta.ton_per_unit.reindex(active)


def run_config(order, bt, origins, scale, resid, cls, method, alpha, kind, policy, lead,
               lead_noise=None):
    """回傳逐 SKU 結果（DataFrame）。lead_noise：callable(sku) → 每日實際前置期陣列。"""
    meas = origins[WARMUP:]
    day0 = meas[0]
    end = bt.date.max()
    days = order.loc[day0:end].index
    rel = {o: days.get_loc(o) for o in meas}
    horizon = POLICIES[policy] + lead
    sub = bt[bt.origin.isin(meas)].set_index(["origin", "sku", "h"])[method].sort_index()
    rows = []
    for sku in order.columns:
        fc = {rel[o]: sub.loc[(o, sku)].sort_index().to_numpy() for o in meas}
        ss = {}
        for o in meas:
            pool = S.past_pool(resid[method], o, cls.index[cls == cls[sku]])
            q = S.safety_factor(pool, alpha, kind)
            ss[rel[o]] = q * scale.loc[o, sku] * np.sqrt(horizon / 7)
        dem = order.loc[day0:end, sku].to_numpy()
        al = lead_noise(sku, len(dem)) if lead_noise else None
        r = S.simulate(dem, fc, ss, policy=policy, lead=lead, actual_lead=al)
        rows.append({"sku": sku, "cls": cls[sku], "demand": r.demand, "filled": r.filled_same_day,
                     "mean_on_hand": r.on_hand.mean(), "mean_demand": dem[1:].mean(),
                     "cycles": len(r.cycle_stockout), "cycles_ok": len(r.cycle_stockout) - sum(r.cycle_stockout),
                     "stockout_days": int((r.unmet_today > S.EPS).sum()), "days": len(dem) - 1,
                     "orders": r.n_orders})
    return pd.DataFrame(rows).set_index("sku")


def summarise(df: pd.DataFrame, tpu: pd.Series) -> dict:
    w = tpu.reindex(df.index)
    out = {"fill_rate": (df.filled * w).sum() / (df.demand * w).sum(),
           "csl": df.cycles_ok.sum() / df.cycles.sum(),
           "days_of_cover": (df.mean_on_hand * w).sum() / (df.mean_demand * w).sum(),
           "stockout_day_share": df.stockout_days.sum() / df.days.sum()}
    for c in ("X", "YZ"):
        d = df[df.cls == c]
        out[f"csl_{c}"] = d.cycles_ok.sum() / d.cycles.sum()
        out[f"fill_rate_{c}"] = (d.filled * w[d.index]).sum() / (d.demand * w[d.index]).sum()
    return out


def cover_at_fill(curve: pd.DataFrame, target: float = FILL_TARGET) -> float:
    """沿著目標 α 的曲線，內插出 fill rate 剛好達到 target 時的庫存天數；達不到回 NaN。"""
    c = curve.sort_values("fill_rate")
    if c.fill_rate.max() < target or c.fill_rate.min() > target:
        return float("nan")
    return float(np.interp(target, c.fill_rate, c.days_of_cover))


def main() -> None:
    order, bt, origins, scale, resid, cls, tpu = build_inputs()

    configs = [dict(method=m, alpha=a, kind=k, policy=p, lead=2)
               for m in F.METHODS for a in ALPHAS for k in KINDS for p in POLICIES]
    configs += [dict(method=m, alpha=a, kind=k, policy=p, lead=L)
                for m in LEAD_SWEEP_METHODS for a in ALPHAS for k in KINDS for p in POLICIES
                for L in LEADS if L != 2]
    per_sku, grid = {}, []
    for c in configs:
        df = run_config(order, bt, origins, scale, resid, cls, **c)
        per_sku[tuple(c.values())] = df
        grid.append({**c, **summarise(df, tpu)})
    grid = pd.DataFrame(grid)
    grid.round(4).to_csv(RESULTS / "m3_grid.csv", index=False)

    def sel(**kw):
        m = np.ones(len(grid), bool)
        for k, v in kw.items():
            m &= grid[k] == v
        return grid[m]

    base = sel(method=BASE["method"], policy="RS", kind="empirical", lead=2).set_index("alpha")

    # ── H1：換目標 vs 換模型 ──
    at95 = sel(alpha=0.95, policy="RS", kind="empirical", lead=2).set_index("method")
    cover_eq = {m: cover_at_fill(sel(method=m, policy="RS", kind="empirical", lead=2))
                for m in F.METHODS}

    def h1_stats(idx):
        def summ(m, a):
            return summarise(per_sku[(m, a, "empirical", "RS", 2)].loc[idx], tpu)
        t_lo, t_hi = summ(BASE["method"], 0.80), summ(BASE["method"], 0.99)
        target_cover = t_hi["days_of_cover"] - t_lo["days_of_cover"]
        target_fill = t_hi["fill_rate"] - t_lo["fill_rate"]
        covers = []
        for m in REASONABLE:
            curve = pd.DataFrame([summ(m, a) for a in ALPHAS])
            covers.append(cover_at_fill(curve))
        covers = np.array(covers)
        # 任何一個方法內插不到就整個 replicate 記 NaN，不拿剩下的方法湊範圍
        method_cover = covers.max() - covers.min() if np.isfinite(covers).all() else np.nan
        return target_cover, target_fill, method_cover

    rng = np.random.default_rng(0)
    skus = np.array(order.columns)
    point = h1_stats(skus)
    boots = np.array([h1_stats(rng.choice(skus, len(skus), replace=True)) for _ in range(N_BOOT)])
    valid = np.isfinite(boots).all(axis=1)
    vb = boots[valid]
    ci = lambda col: [_r(np.percentile(vb[:, col], 2.5), 2), _r(np.percentile(vb[:, col], 97.5), 2)]
    diff = vb[:, 0] - vb[:, 2]
    h1 = {
        "base": BASE,
        "target_effect_0.80_to_0.99": {"days_of_cover": _r(point[0], 2), "ci95": ci(0),
                                        "fill_rate": _r(point[1]), "fill_rate_ci95": ci(1)},
        "fill_target_for_equal_service": FILL_TARGET,
        "bootstrap_valid_share": _r(valid.mean(), 3),
        "method_effect_cover_at_equal_fill": {"range_days": _r(point[2], 2), "ci95": ci(2),
                                             "methods": REASONABLE,
                                             "cover_by_method": {m: _r(v, 2) for m, v in cover_eq.items()}},
        "target_minus_method_days": {"point": _r(point[0] - point[2], 2),
                                     "ci95": [_r(np.percentile(diff, 2.5), 2),
                                              _r(np.percentile(diff, 97.5), 2)],
                                     "share_boot_positive": _r(np.mean(diff > 0), 3)},
        "at_alpha_0.95_by_method": {m: {k: _r(at95.loc[m, k]) for k in ("fill_rate", "csl", "days_of_cover")}
                                    for m in F.METHODS},
    }
    h1["supported"] = bool(valid.mean() >= 0.9 and h1["target_minus_method_days"]["ci95"][0] > 0)

    # ── H2：常態近似 vs 經驗分位數，X vs Y＋Z ──
    h2_rows = {}
    for kind in KINDS:
        g = sel(method=BASE["method"], policy="RS", kind=kind, lead=2).set_index("alpha")
        h2_rows[kind] = {str(a): {"csl_X": _r(g.loc[a, "csl_X"]), "csl_YZ": _r(g.loc[a, "csl_YZ"]),
                                  "shortfall_X": _r(g.loc[a, "csl_X"] - a),
                                  "shortfall_YZ": _r(g.loc[a, "csl_YZ"] - a)} for a in ALPHAS}
    hi = [a for a in ALPHAS if a >= 0.90]
    mean_sf = {k: {c: _r(np.mean([h2_rows[k][str(a)][f"shortfall_{c}"] for a in hi]))
                   for c in ("X", "YZ")} for k in KINDS}
    h2 = {"by_kind": h2_rows, "mean_shortfall_alpha_ge_0.90": mean_sf,
          "supported": bool(mean_sf["normal"]["YZ"] < mean_sf["normal"]["X"] - 0.02
                            and mean_sf["empirical"]["YZ"] > mean_sf["normal"]["YZ"])}

    # ── 前置期敏感度（政策參數按名目 L 算）──
    lead_curve = {str(L): {k: _r(v) for k, v in sel(method=BASE["method"], policy="RS", kind="empirical",
                                                     lead=L, alpha=0.95).iloc[0][
        ["fill_rate", "csl", "days_of_cover"]].items()} for L in LEADS}

    # ── Monte Carlo：實際前置期 = L + U{-1,0,+1} ──
    mc = {}
    for a in ALPHAS:
        vals = []
        for seed in range(N_MC):
            g = np.random.default_rng(seed)
            noise = lambda sku, n, g=g: np.clip(2 + g.integers(-1, 2, n), 1, None)
            df = run_config(order, bt, origins, scale, resid, cls, BASE["method"], a, "empirical",
                            "RS", 2, lead_noise=noise)
            s = summarise(df, tpu)
            vals.append((s["fill_rate"], s["days_of_cover"], s["csl"]))
        v = np.array(vals)
        mc[str(a)] = {k: {"p05": _r(np.percentile(v[:, i], 5)), "p50": _r(np.percentile(v[:, i], 50)),
                          "p95": _r(np.percentile(v[:, i], 95))}
                      for i, k in enumerate(("fill_rate", "days_of_cover", "csl"))}

    out = {
        "design": {"measured_origins": [str(pd.Timestamp(o).date()) for o in origins[WARMUP:]],
                   "warmup_origins": WARMUP, "alphas": ALPHAS, "kinds": KINDS,
                   "policies": list(POLICIES), "leads": LEADS, "n_configs": len(configs),
                   "n_boot": N_BOOT, "n_mc": N_MC, "skus": len(skus),
                   "class_sizes": cls.value_counts().to_dict()},
        "base_curve": {str(a): {k: _r(base.loc[a, k]) for k in ("fill_rate", "csl", "days_of_cover",
                                                                 "stockout_day_share")} for a in ALPHAS},
        "h1": h1, "h2": h2, "lead_time_sensitivity_alpha_0.95": lead_curve, "monte_carlo_lead_noise": mc,
    }
    (RESULTS / "m3_policy.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("base_curve", "lead_time_sensitivity_alpha_0.95")}, indent=1))
    print(json.dumps({"h1": {k: v for k, v in h1.items() if k != "at_alpha_0.95_by_method"},
                      "h2": h2["mean_shortfall_alpha_ge_0.90"], "h2_supported": h2["supported"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
