"""KPI 層：全部是純函式，輸入序列、輸出數字，不碰檔案。

這份資料沒有逐單交期、沒有期初庫存、沒有期初欠貨。所以每個 KPI 都要回答兩件事：
定義是什麼，以及它在這份資料上的限制是什麼（見 openspec design D4）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-9


# ── fill rate ──────────────────────────────────────────────────────────
def fill_rate(order, delivery) -> float:
    """期間 in-full 比率：min(Σ出貨, Σ訂單) ÷ Σ訂單。

    出貨多於訂單的部分是在消化期初欠貨，不能把比率推過 1。沒有訂單時回 NaN。
    """
    o, d = float(np.sum(order)), float(np.sum(delivery))
    if o <= EPS:
        return float("nan")
    return min(d, o) / o


def fill_rate_weighted(order: pd.DataFrame, delivery: pd.DataFrame, weight: pd.Series) -> float:
    """以噸數加權的組合 fill rate：Σ_sku min(D, O)·w ÷ Σ_sku O·w。"""
    o, d = order.sum(), delivery.sum()
    w = weight.reindex(o.index)
    return float((np.minimum(o, d) * w).sum() / (o * w).sum())


# ── FIFO 欠貨帳齡 ──────────────────────────────────────────────────────
@dataclass
class FifoResult:
    lots: pd.DataFrame            # day, qty, filled_day（未交齊為 NaN）, days_to_fill
    oldest_open_age: np.ndarray   # 每天結束時最老未交齊批的年齡（天）；沒有未交齊批為 NaN
    unmatched_delivery: float     # 沒有任何未交齊訂單可配的出貨量
    open_at_end: float            # 期末仍未交齊的量（不含期初欠貨）
    open_qty: np.ndarray | None = None   # 每天結束時未交齊的總量（含期初欠貨批）


def fifo(order, delivery, opening_backlog: float = 0.0) -> FifoResult:
    """先進先出：每天先把當日訂單排進佇列，再用當日出貨從最老的批開始沖銷。

    期初欠貨（若給）視為第 -1 天的一批，排在最前面，但不計入 KPI 的訂單批。
    這個假設的方向是保守的：期初有舊單時，新單的交付一定被往後推。
    """
    order = np.asarray(order, dtype=float)
    delivery = np.asarray(delivery, dtype=float)
    n = len(order)
    queue: list[list[float]] = []   # [day, remaining, lot_index]
    if opening_backlog > EPS:
        queue.append([-1, float(opening_backlog), -1])

    lot_day, lot_qty, filled = [], [], []
    oldest = np.full(n, np.nan)
    open_qty = np.zeros(n)
    unmatched = 0.0
    for t in range(n):
        if order[t] > EPS:
            queue.append([t, order[t], len(lot_day)])
            lot_day.append(t)
            lot_qty.append(order[t])
            filled.append(np.nan)
        left = delivery[t]
        while left > EPS and queue:
            head = queue[0]
            take = min(left, head[1])
            head[1] -= take
            left -= take
            if head[1] <= EPS:
                if head[2] >= 0:
                    filled[head[2]] = t
                queue.pop(0)
        unmatched += max(left, 0.0)
        if queue:
            oldest[t] = t - queue[0][0]
            open_qty[t] = sum(q[1] for q in queue)

    lots = pd.DataFrame({"day": lot_day, "qty": lot_qty, "filled_day": filled})
    lots["days_to_fill"] = lots.filled_day - lots.day
    open_end = float(sum(q[1] for q in queue if q[2] >= 0))
    return FifoResult(lots, oldest, unmatched, open_end, open_qty)


def otif_curve(lots: pd.DataFrame, max_days: int = 7) -> dict:
    """D = 0…max_days 時，在 D 天內交齊的訂單批比例（批數口徑與量口徑各一）。"""
    if lots.empty:
        return {}
    out = {}
    for d in range(max_days + 1):
        ok = lots.days_to_fill <= d          # NaN（未交齊）比較結果為 False
        out[d] = {"by_lot": float(ok.mean()),
                  "by_qty": float(lots.qty[ok].sum() / lots.qty.sum())}
    return out


def stockout_flags(oldest_open_age: np.ndarray, max_days: int) -> np.ndarray:
    """某天結束時，若有訂單批已等超過 max_days 天仍未交齊，該產品日記為缺貨。"""
    age = np.nan_to_num(oldest_open_age, nan=-1)
    return age > max_days


# ── 庫存 ──────────────────────────────────────────────────────────────
def opening_lower_bound(inflow, outflow) -> float:
    """讓「期初 + 累計流入 − 累計流出」永不為負的最小期初值。"""
    net = np.cumsum(np.asarray(inflow, float)) - np.cumsum(np.asarray(outflow, float))
    return float(max(0.0, -net.min())) if len(net) else 0.0


def stock_level(inflow, outflow, opening: float) -> np.ndarray:
    return opening + np.cumsum(np.asarray(inflow, float)) - np.cumsum(np.asarray(outflow, float))


def days_of_cover(stock, demand, window: int = 28) -> np.ndarray:
    """庫存 ÷ 近 window 天平均日需求。近期無需求時回 NaN（不是無限大）。"""
    avg = pd.Series(np.asarray(demand, float)).rolling(window, min_periods=7).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cover = np.asarray(stock, float) / avg
    cover[~(avg > EPS)] = np.nan
    return cover


# ── ABC × XYZ ─────────────────────────────────────────────────────────
def abc_class(volume: pd.Series, a: float = 0.80, b: float = 0.95) -> pd.Series:
    """依量由大到小累計；在跨過 a 之前開始的品項屬 A，跨過 b 之前開始的屬 B，其餘 C。"""
    v = volume.sort_values(ascending=False)
    share_before = v.cumsum().shift(fill_value=0) / v.sum()
    cls = np.where(share_before < a, "A", np.where(share_before < b, "B", "C"))
    return pd.Series(cls, index=v.index).reindex(volume.index)


def xyz_class(daily: pd.DataFrame, x: float = 0.5, y: float = 1.0) -> pd.Series:
    """週需求的變異係數；只用完整的週，避開頭尾殘週。用週不用日，避開週末為 0 的假波動。"""
    weeks = daily.resample("W-SAT")
    counts = weeks.size()
    weekly = weeks.sum()[counts == 7]
    mean = weekly.mean()
    cv = weekly.std(ddof=0) / mean.where(mean > EPS)
    cls = np.where(cv < x, "X", np.where(cv < y, "Y", "Z"))
    return pd.Series(cls, index=daily.columns).where(cv.notna(), "Z")


def segment(order_ton: pd.DataFrame, dormant: list[str]) -> pd.DataFrame:
    """ABC 用噸數（不同包裝規格才能比），XYZ 用週需求 CV；休眠 SKU 標 D。"""
    active = [c for c in order_ton.columns if c not in set(dormant)]
    abc = abc_class(order_ton[active].sum())
    xyz = xyz_class(order_ton[active])
    seg = pd.DataFrame({"abc": abc, "xyz": xyz})
    seg["segment"] = seg.abc + seg.xyz
    dorm = pd.DataFrame({"abc": "D", "xyz": "D", "segment": "D"}, index=list(dormant))
    return pd.concat([seg, dorm]).reindex(order_ton.columns)


# ── 預測誤差（M2 用）──────────────────────────────────────────────────
def wape(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = np.abs(a).sum()
    return float(np.abs(f - a).sum() / denom) if denom > EPS else float("nan")


def bias(actual, forecast) -> float:
    """正值＝高估，負值＝低估。"""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = np.abs(a).sum()
    return float((f - a).sum() / denom) if denom > EPS else float("nan")
