"""品質檢核：只報告、不修資料。

分兩類：
- 硬錯誤（`HardError`）：日期不連續、負值、噸／件比不固定。遇到就中止，因為後面的
  KPI 全部建立在「每天一列、量非負、兩種單位是同一份資料」上。
- 觀察（寫進報告）：星期分布、出貨超前訂單、期初庫存與期初欠貨的下限、休眠 SKU、
  共用代碼的 SKU、缺廠別對照的 SKU。這些多半是資訊不足，不是錯誤；修掉等於把限制藏起來。
"""
from __future__ import annotations

import pandas as pd

import kpi
from data import pivot

DORMANT_ZERO_SHARE = 0.98
TON_RATIO_TOL = 1e-6


class HardError(ValueError):
    pass


def check_dates(wide: pd.DataFrame) -> dict:
    idx = wide.index
    gaps = int((pd.Series(idx).diff().dt.days > 1).sum())
    dups = int(idx.duplicated().sum())
    if gaps or dups or not idx.is_monotonic_increasing:
        raise HardError(f"日期不連續：缺口 {gaps} 處、重複 {dups} 處")
    return {"start": str(idx.min().date()), "end": str(idx.max().date()), "days": len(idx)}


def check_non_negative(flows: pd.DataFrame) -> dict:
    neg = flows[(flows.qty_unit < 0) | (flows.qty_ton < 0)]
    if len(neg):
        raise HardError(f"出現負值 {len(neg)} 列，例：{neg.iloc[0].to_dict()}")
    return {"negative_rows": 0}


def check_ton_ratio(flows: pd.DataFrame) -> dict:
    """上游噸數版（qty_ton_reported）與件數版是不是同一份資料。

    硬錯誤：同一 SKU 的噸／件比不固定；件數 0 但噸數非 0；同一條（SKU, 流量）裡
    有的列有噸數、有的列沒有。
    容許但要報告：某條（SKU, 流量）的噸數整欄為 0——上游缺欄，噸數改由件數 × 係數重算。
    """
    rep = flows.qty_ton_reported
    both = flows[(flows.qty_unit > 0) & (rep > 0)]
    ratio = both.qty_ton_reported / both.qty_unit
    spread = ratio.groupby(both.sku).agg(lambda r: (r.max() - r.min()) / r.mean())
    bad = spread[spread > TON_RATIO_TOL]
    ghost = int(((flows.qty_unit == 0) & (rep != 0)).sum())

    pos = flows[flows.qty_unit > 0]
    has_ton = (pos.qty_ton_reported > 0).groupby([pos.sku, pos.flow]).agg(["all", "any"])
    partial = has_ton[has_ton["any"] & ~has_ton["all"]]
    missing = has_ton[~has_ton["any"]]
    if len(bad) or ghost or len(partial):
        raise HardError(f"噸／件比不固定的 SKU：{list(bad.index)[:5]}；件數 0 但噸數非 0：{ghost} 列；"
                        f"部分缺噸數：{[f'{s}/{f}' for s, f in partial.index][:5]}")
    return {"max_relative_spread": float(spread.max()),
            "weight_column_missing": [f"{s}/{f}" for s, f in missing.index]}


def weekday_profile(order: pd.DataFrame) -> dict:
    """各星期的平均日訂單量（全 SKU 加總）。0 = 週一。"""
    daily = order.sum(axis=1)
    prof = daily.groupby(daily.index.dayofweek).mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {names[i]: round(float(v), 1) for i, v in prof.items()}


def opening_lower_bound(inflow: pd.DataFrame, outflow: pd.DataFrame) -> pd.Series:
    """逐 SKU 的期初下限。用在倉庫庫存（流入＝工廠出庫）與期初欠貨（流入＝訂單）。"""
    return pd.Series({c: kpi.opening_lower_bound(inflow[c], outflow[c]) for c in inflow.columns})


def dormant_skus(order: pd.DataFrame, threshold: float = DORMANT_ZERO_SHARE) -> list[str]:
    zero_share = (order == 0).mean()
    return sorted(zero_share[zero_share >= threshold].index)


def report(flows: pd.DataFrame, skus: pd.DataFrame, sites: pd.DataFrame) -> dict:
    order, delivery = pivot(flows, "order"), pivot(flows, "delivery")
    issue = pivot(flows, "factory_issue")

    out = {
        "dates": check_dates(order),
        "non_negative": check_non_negative(flows),
        "ton_ratio": check_ton_ratio(flows),
        "n_skus": int(order.shape[1]),
        "weekday_mean_order_units": weekday_profile(order),
    }
    cum_gap = delivery.cumsum() - order.cumsum()
    out["delivery_ahead_of_order_share"] = round(float((cum_gap > 1e-9).values.mean()), 4)

    ob = opening_lower_bound(order, delivery)
    os_ = opening_lower_bound(issue, delivery)
    out["opening_backlog_lower_bound_units"] = {
        "skus_positive": int((ob > 0).sum()), "total": round(float(ob.sum()), 1)}
    out["opening_stock_lower_bound_units"] = {
        "skus_positive": int((os_ > 0).sum()), "total": round(float(os_.sum()), 1)}

    d = dormant_skus(order)
    out["dormant_skus"] = {"threshold_zero_share": DORMANT_ZERO_SHARE, "count": len(d), "skus": d}
    out["shared_code_skus"] = sorted(s for s in order.columns if s.endswith(".1"))
    out["skus_without_site_mapping"] = sorted(set(order.columns) - set(sites.sku))
    tot = flows.groupby("flow").qty_ton.sum()
    out["total_tons"] = {k: round(float(v), 1) for k, v in tot.items()}
    return out
