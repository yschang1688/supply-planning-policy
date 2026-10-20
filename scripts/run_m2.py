"""M2：15 個起點的 rolling-origin 回測，落盤 results/m2_forecast.json 與 m2_backtest.csv。

    python scripts/run_m2.py      # 需先跑過 run_m0_m1.py（讀 m0_quality.json 與 m1_sku_table.csv）
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import data  # noqa: E402
import forecast as F  # noqa: E402
import kpi  # noqa: E402

RESULTS = ROOT / "results"
REFERENCE = "weekday_mean"      # 計畫員手算法；其他方法都跟它比


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def score(df: pd.DataFrame, methods) -> dict:
    """噸數口徑的 WAPE 與 bias。"""
    return {m: {"wape": _r(kpi.wape(df.actual_t, df[m + "_t"])),
                "bias": _r(kpi.bias(df.actual_t, df[m + "_t"]))} for m in methods}


def paired(per_origin: pd.DataFrame, a: str, b: str) -> dict:
    """15 個起點的成對 WAPE 差（a − b，負值＝a 較好）。"""
    d = (per_origin[a] - per_origin[b]).dropna()
    p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else float("nan")
    return {"median_diff": _r(d.median()), "a_wins": int((d < 0).sum()), "n": int(len(d)),
            "wilcoxon_p": _r(p)}


def main() -> None:
    flows, skus, _ = data.load()
    q = json.loads((RESULTS / "m0_quality.json").read_text(encoding="utf-8"))
    dormant = set(q["dormant_skus"]["skus"])
    order = data.pivot(flows, "order")
    order = order[[c for c in order.columns if c not in dormant]]
    meta = skus.set_index("sku")
    seg = pd.read_csv(RESULTS / "m1_sku_table.csv", index_col="sku")["segment"]

    idx = F.origins(len(order))
    bt = F.backtest(order, meta.group, idx)
    tpu = bt.sku.map(meta.ton_per_unit)
    bt["actual_t"] = bt.actual * tpu
    for m in F.METHODS:
        bt[m + "_t"] = bt[m] * tpu
    bt["segment"] = bt.sku.map(seg)
    eid = F.eid_days()
    holiday_origins = set(bt.loc[bt.date.isin(eid), "origin"])
    bt["holiday_window"] = bt.origin.isin(holiday_origins)

    tcols = ["actual_t"] + [m + "_t" for m in F.METHODS]
    weekly = bt.groupby(["origin", "sku", "holiday_window"], as_index=False)[tcols].sum()

    per_origin = pd.DataFrame({m: bt.groupby("origin").apply(
        lambda g, m=m: kpi.wape(g.actual_t, g[m + "_t"]), include_groups=False) for m in F.METHODS})
    clean = per_origin[~per_origin.index.isin(holiday_origins)]

    out = {
        "design": {"n_origins": len(idx), "horizon_days": F.H, "skus": order.shape[1],
                   "first_origin": str(order.index[idx[0]].date()),
                   "last_origin": str(order.index[idx[-1]].date()),
                   "holiday_origins": sorted(str(d.date()) for d in holiday_origins),
                   "reference_method": REFERENCE},
        "overall": score(bt, F.METHODS),
        "excluding_holiday_windows": score(bt[~bt.holiday_window], F.METHODS),
        # 計畫的時間桶是週：同一起點、同一 SKU 的 7 天加總後再算誤差
        "weekly_bucket": score(weekly, F.METHODS),
        "weekly_bucket_excluding_holiday": score(weekly[~weekly.holiday_window], F.METHODS),
        "by_segment": {s: score(g, F.METHODS) for s, g in bt.groupby("segment")},
        "by_abc": {a: score(g, F.METHODS) for a, g in bt.groupby(bt.segment.str[0])},
        "by_horizon": {int(h): score(g, F.METHODS) for h, g in bt.groupby("h")},
        "per_origin_wape": {str(k.date()): {m: _r(v) for m, v in row.items()}
                            for k, row in per_origin.iterrows()},
        "paired_vs_reference": {
            "all_origins": {m: paired(per_origin, m, REFERENCE) for m in F.METHODS if m != REFERENCE},
            "excluding_holiday_origins": {m: paired(clean, m, REFERENCE)
                                          for m in F.METHODS if m != REFERENCE},
        },
        "calendar_value": {
            "all_origins": paired(per_origin, "lgbm_calendar", "lgbm"),
            "excluding_holiday_origins": paired(clean, "lgbm_calendar", "lgbm"),
        },
    }
    best = min(F.METHODS, key=lambda m: out["overall"][m]["wape"])
    out["best_overall"] = best
    (RESULTS / "m2_forecast.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    cols = ["origin", "date", "sku", "h", "segment", "holiday_window", "actual", *F.METHODS]
    bt[cols].round(3).to_csv(RESULTS / "m2_backtest.csv", index=False)
    print(json.dumps({k: out[k] for k in ("design", "overall", "excluding_holiday_windows", "weekly_bucket",
                                          "by_abc", "paired_vs_reference", "calendar_value",
                                          "best_overall")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
