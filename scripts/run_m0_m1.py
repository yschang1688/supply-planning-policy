"""M0＋M1：下載（或驗證）資料 → 品質報告 → KPI 層，落盤到 results/。

    python scripts/run_m0_m1.py
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
import quality  # noqa: E402

RESULTS = ROOT / "results"
MAX_D = 7
STOCKOUT_D = 7   # 等超過一週仍未交齊才算缺貨：保守口徑，曲線另見 stockout_rate_by_d


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def fifo_summary(order: pd.DataFrame, delivery: pd.DataFrame, tpu: pd.Series,
                 skus: list[str], opening: pd.Series) -> tuple[dict, pd.DataFrame]:
    """跑所有 SKU 的 FIFO，彙總成組合層 KPI（批數口徑與噸數口徑），並回傳逐 SKU 摘要。"""
    all_lots, flags, rows = [], [], []
    unmatched_t = open_end_t = 0.0
    for s in skus:
        r = kpi.fifo(order[s].to_numpy(), delivery[s].to_numpy(), opening_backlog=opening[s])
        lots = r.lots.assign(sku=s, qty_ton=r.lots.qty * tpu[s])
        all_lots.append(lots)
        flags.append(np.stack([kpi.stockout_flags(r.oldest_open_age, d) for d in range(MAX_D + 1)]))
        unmatched_t += r.unmatched_delivery * tpu[s]
        open_end_t += r.open_at_end * tpu[s]
        dtf = lots.days_to_fill
        rows.append({"sku": s,
                     "median_days_to_fill": dtf.median(),
                     "p90_days_to_fill": dtf.quantile(0.9),
                     "share_lots_filled_within_7d": (dtf <= 7).mean(),
                     "stockout_rate_d7": flags[-1][STOCKOUT_D].mean()})
    lots = pd.concat(all_lots, ignore_index=True)
    flag = np.stack(flags)                       # sku × D × day
    ordered_t = lots.qty_ton.sum()
    summary = {
        "n_lots": int(len(lots)),
        "otif_proxy_by_d": {d: {"by_lot": _r((lots.days_to_fill <= d).mean()),
                                "by_ton": _r(lots.qty_ton[lots.days_to_fill <= d].sum() / ordered_t)}
                            for d in range(MAX_D + 1)},
        "stockout_rate_by_d": {d: _r(flag[:, d, :].mean()) for d in range(MAX_D + 1)},
        "median_days_to_fill": _r(lots.days_to_fill.median(), 1),
        "p90_days_to_fill": _r(lots.days_to_fill.quantile(0.9), 1),
        "share_lots_never_filled": _r(lots.filled_day.isna().mean()),
        "unfilled_at_end_tons": _r(open_end_t, 1),
        "unmatched_delivery_tons": _r(unmatched_t, 1),
    }
    return summary, pd.DataFrame(rows).set_index("sku")


def main() -> None:
    data.download()
    flows, skus, sites = data.load()
    RESULTS.mkdir(exist_ok=True)

    q = quality.report(flows, skus, sites)
    (RESULTS / "m0_quality.json").write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")

    order, delivery = data.pivot(flows, "order"), data.pivot(flows, "delivery")
    issue = data.pivot(flows, "factory_issue")
    order_t = data.pivot(flows, "order", "qty_ton")
    tpu = skus.set_index("sku").ton_per_unit
    dormant = q["dormant_skus"]["skus"]
    active = [s for s in order.columns if s not in set(dormant)]

    seg = kpi.segment(order_t, dormant)
    backlog0 = pd.Series({s: kpi.opening_lower_bound(order[s], delivery[s]) for s in order.columns})
    stock0 = pd.Series({s: kpi.opening_lower_bound(issue[s], delivery[s]) for s in order.columns})

    zero = pd.Series(0.0, index=order.columns)
    fifo_a, sku_a = fifo_summary(order, delivery, tpu, active, zero)
    fifo_b, sku_b = fifo_summary(order, delivery, tpu, active, backlog0)

    # 庫存天數（下限）：組合層用噸數加總
    stock_t = pd.DataFrame({s: kpi.stock_level(issue[s], delivery[s], stock0[s]) * tpu[s]
                            for s in active}, index=order.index)
    cover_port = kpi.days_of_cover(stock_t.sum(axis=1), order_t[active].sum(axis=1))
    cover_sku = pd.Series({s: np.nanmedian(kpi.days_of_cover(stock_t[s], order_t[s]))
                           for s in active})

    out = {
        "scope": {"active_skus": len(active), "dormant_skus": len(dormant),
                  "stockout_definition_days": STOCKOUT_D},
        "fill_rate_ton_weighted": _r(kpi.fill_rate_weighted(order[active], delivery[active], tpu)),
        "fifo_opening_backlog_zero": fifo_a,
        "fifo_opening_backlog_lower_bound": fifo_b,
        "days_of_cover_lower_bound": {
            "portfolio_median": _r(np.nanmedian(cover_port), 1),
            "portfolio_p10": _r(np.nanpercentile(cover_port, 10), 1),
            "portfolio_p90": _r(np.nanpercentile(cover_port, 90), 1),
            "sku_median_of_medians": _r(cover_sku.median(), 1),
        },
        "segments": seg.segment.value_counts().sort_index().to_dict(),
    }
    (RESULTS / "m1_kpi.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    table = (skus.set_index("sku")[["group", "subgroup", "ton_per_unit"]]
             .join(seg)
             .join(order_t.sum().rename("order_tons"))
             .join(pd.Series({s: kpi.fill_rate(order[s], delivery[s]) for s in order.columns},
                             name="fill_rate"))
             .join(sku_b)
             .join(cover_sku.rename("median_days_of_cover_lb"))
             .sort_values("order_tons", ascending=False))
    table.round(4).to_csv(RESULTS / "m1_sku_table.csv")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
