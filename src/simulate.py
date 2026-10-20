"""M3 庫存政策模擬：逐日重播實際需求，比較 (R,S) 與 (s,S) 在不同安全庫存下的表現。

時間約定：第 0 天＝第一個預測起點（日終下第一張單），需求從第 1 天開始。
`forecasts[o]` 是在第 o 天日終做的預測，長度 7，對應第 o+1…o+7 天；
超過 7 天的需求沿用同一週的星期型態（循環）。`ss[o]` 是第 o 天起適用的安全庫存（件）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

EPS = 1e-9


# ── 安全係數 q ────────────────────────────────────────────────────────
def safety_factor(r_past: np.ndarray, alpha: float, kind: str) -> float:
    """r_past：過去起點的週相對誤差（實際 − 預測）÷ 尺度。回傳 q，SS = q × 尺度 × √(期間/7)。

    normal_classic：z·σ（假設無偏）；normal：μ + z·σ；empirical：α 分位數。
    q 不得為負：預測系統性高估時，安全庫存歸零，不是負的。
    """
    r = np.asarray(r_past, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        raise ValueError("過去誤差不足 2 個，無法估安全係數")
    z = NormalDist().inv_cdf(alpha)
    if kind == "normal_classic":
        q = z * r.std(ddof=1)
    elif kind == "normal":
        q = r.mean() + z * r.std(ddof=1)
    elif kind == "empirical":
        q = float(np.quantile(r, alpha, method="higher"))
    else:
        raise ValueError(kind)
    return max(0.0, float(q))


def past_pool(resid, cutoff, columns) -> np.ndarray:
    """起點 cutoff 可用的誤差池：只取 cutoff **之前**的起點（cutoff 當天那一列是它之後一週
    的誤差，屬於未來），以及同一群的 SKU 欄。resid：DataFrame（index＝起點、columns＝SKU）。"""
    return resid.loc[resid.index < cutoff, columns].to_numpy().ravel()


# ── 模擬引擎 ──────────────────────────────────────────────────────────
@dataclass
class SimResult:
    demand: float
    filled_same_day: float
    shipped: float
    end_backlog: float
    initial_on_hand: float
    received: float
    end_on_hand: float
    on_hand: np.ndarray                 # 第 1…T 天日終在手
    unmet_today: np.ndarray             # 第 1…T 天當天沒出的量
    cycle_stockout: list = field(default_factory=list)   # 每個 (R,S) 週期是否缺過貨
    n_orders: int = 0

    @property
    def fill_rate(self) -> float:
        return self.filled_same_day / self.demand if self.demand > EPS else float("nan")


def _fc(forecasts: dict, o: int, t: int, k: int) -> float:
    """第 o 天的預測裡，第 t+k 天那一格（t ≥ o，k ≥ 1）。"""
    return float(forecasts[o][(t + k - o - 1) % 7])


def simulate(demand: np.ndarray, forecasts: dict[int, np.ndarray], ss: dict[int, float],
             policy: str = "RS", lead: int = 2, review: int = 7,
             actual_lead: np.ndarray | None = None) -> SimResult:
    """demand[t] 是第 t 天的需求（demand[0] 不使用，第 0 天只下單）。

    actual_lead 給定時，第 t 天下的單在 t + actual_lead[t] 到貨（Monte Carlo 用）；
    否則一律 t + lead。政策參數永遠用名目 lead 計算。
    """
    demand = np.asarray(demand, float)
    T = len(demand) - 1
    origins = sorted(forecasts)
    if origins[0] != 0:
        raise ValueError("第一個預測起點必須是第 0 天")
    horizon = review + lead if policy == "RS" else lead

    def latest(t):
        return max(o for o in origins if o <= t)

    def order_up_to(t):
        o = latest(t)
        if policy == "RS":
            S = sum(_fc(forecasts, o, t, k) for k in range(1, review + lead + 1)) + ss[o]
            return S, S
        s = sum(_fc(forecasts, o, t, k) for k in range(1, lead + 1)) + ss[o]
        S = s + sum(_fc(forecasts, o, t, lead + k) for k in range(1, 8))
        return s, S

    _, S0 = order_up_to(0)
    on_hand = S0
    initial = S0
    arrivals = np.zeros(T + 1 + 64)
    backlog: list[list[float]] = []
    pipeline = 0.0
    filled_same = shipped = received = 0.0
    oh = np.zeros(T)
    unmet = np.zeros(T)
    stockout_day = np.zeros(T + 1, bool)
    n_orders = 0

    def place(t):
        nonlocal pipeline, n_orders
        s, S = order_up_to(t)
        ip = on_hand + pipeline - sum(b[1] for b in backlog)
        if policy == "RS" or ip <= s + EPS:
            q = max(0.0, S - ip)
            if q > EPS:
                L = int(actual_lead[t]) if actual_lead is not None else lead
                arrivals[t + max(1, L)] += q
                pipeline += q
                n_orders += 1

    place(0)
    for t in range(1, T + 1):
        got = arrivals[t]
        on_hand += got
        pipeline -= got
        received += got
        # 先補舊欠單（先進先出），再出今天的單
        while backlog and on_hand > EPS:
            take = min(on_hand, backlog[0][1])
            on_hand -= take
            shipped += take
            backlog[0][1] -= take
            if backlog[0][1] <= EPS:
                backlog.pop(0)
        today = min(on_hand, demand[t])
        on_hand -= today
        shipped += today
        filled_same += today
        short = demand[t] - today
        if short > EPS:
            backlog.append([t, short])
            stockout_day[t] = True
        oh[t - 1] = on_hand
        unmet[t - 1] = short
        if policy == "sS" or t in forecasts:
            place(t)

    cycles = [bool(stockout_day[o + 1: o + review + 1].any()) for o in origins if o + review <= T]
    return SimResult(demand=float(demand[1:].sum()), filled_same_day=filled_same, shipped=shipped,
                     end_backlog=float(sum(b[1] for b in backlog)), initial_on_hand=initial,
                     received=received, end_on_hand=on_hand, on_hand=oh, unmet_today=unmet,
                     cycle_stockout=cycles, n_orders=n_orders)
