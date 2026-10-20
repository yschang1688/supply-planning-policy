"""M4 產能排程：週別、多 SKU、有產能限制的批量問題（CLSP），以 HiGHS 求解。

求解器走 scipy.optimize.milp（內建 HiGHS）。不用 PuLP 的原因：PuLP 4 改成新 API 且不再附 CBC，
PuLP 2.x 附的 CBC 執行檔只有 x86_64，在 Apple Silicon 上跑不起來（2026-10-21 實測）。

變數排列：每個 (SKU i, 週 t) 依序有 x（生產噸）、y（開線 0/1）、I（期末庫存）、B（期末欠單）、
U（低於安全庫存的缺口，只在給 safety 時有成本）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

X, Y, INV, BL, U = range(5)
NV = 5
STATUS = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "other"}


@dataclass
class Plan:
    status: str
    objective: float
    mip_gap: float
    x: np.ndarray        # n × T
    y: np.ndarray
    inv: np.ndarray
    backlog: np.ndarray

    @property
    def setups(self) -> int:
        return int(round(self.y.sum()))


def solve(demand: np.ndarray, groups: list, capacity: dict, min_batch: np.ndarray,
          net0: np.ndarray, setup_cost: float, hold: float = 1.0, penalty: float = 10.0,
          terminal: np.ndarray | None = None, allow_backlog: bool = True,
          safety: np.ndarray | None = None, safety_penalty: float = 2.0,
          time_limit: float = 60.0, gap: float = 0.005) -> Plan:
    """demand：n × T（噸）；groups：每個 SKU 的產線；capacity：產線 → 每週噸；net0：期初淨庫存。

    terminal 給定時要求 I_iT − B_iT ≥ terminal_i。allow_backlog=False 時 B 恆為 0。
    safety（n × T）給定時是軟下限：I_it + U_it ≥ safety_it，缺口 U 每噸週罰 safety_penalty。
    """
    demand = np.asarray(demand, float)
    min_batch = np.asarray(min_batch, float)
    net0 = np.asarray(net0, float)
    n, T = demand.shape
    for name, arr in (("demand", demand), ("min_batch", min_batch), ("net0", net0)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} 含非有限值；先排除沒有生產紀錄的 SKU")
    nv = NV * n * T

    def v(k, i, t):
        return NV * (i * T + t) + k

    c = np.zeros(nv)
    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    integrality = np.zeros(nv)
    for i in range(n):
        cap = capacity[groups[i]]
        for t in range(T):
            c[v(INV, i, t)] = hold
            c[v(BL, i, t)] = penalty
            c[v(Y, i, t)] = setup_cost
            if safety is not None:
                c[v(U, i, t)] = safety_penalty
            else:
                ub[v(U, i, t)] = 0
            ub[v(Y, i, t)] = 1
            integrality[v(Y, i, t)] = 1
            ub[v(X, i, t)] = cap
            if not allow_backlog:
                ub[v(BL, i, t)] = 0

    rows, lo, hi = [], [], []
    A = lil_matrix((n * T * 4 + T * len(capacity) + n, nv))
    r = 0

    def add(coefs, low, high):
        nonlocal r
        for j, a in coefs:
            A[r, j] = a
        lo.append(low)
        hi.append(high)
        r += 1

    for i in range(n):
        cap = capacity[groups[i]]
        for t in range(T):
            # 平衡：I_t − B_t − x_t − (I_{t−1} − B_{t−1}) = −d_t；t=0 時前期淨庫存是常數 net0
            co = [(v(INV, i, t), 1), (v(BL, i, t), -1), (v(X, i, t), -1)]
            rhs = -demand[i, t]
            if t == 0:
                rhs += net0[i]
            else:
                co += [(v(INV, i, t - 1), -1), (v(BL, i, t - 1), 1)]
            add(co, rhs, rhs)
            add([(v(X, i, t), 1), (v(Y, i, t), -cap)], -np.inf, 0)          # 開線連結
            add([(v(Y, i, t), min_batch[i]), (v(X, i, t), -1)], -np.inf, 0)  # 最小批量
            if safety is not None:
                add([(v(INV, i, t), 1), (v(U, i, t), 1)], safety[i, t], np.inf)  # 安全庫存軟下限
    for g, cap in capacity.items():
        members = [i for i in range(n) if groups[i] == g]
        for t in range(T):
            add([(v(X, i, t), 1) for i in members], -np.inf, cap)          # 產線產能
    if terminal is not None:
        for i in range(n):
            add([(v(INV, i, T - 1), 1), (v(BL, i, T - 1), -1)], terminal[i], np.inf)

    A = A[:r].tocsr()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})
    status = STATUS.get(res.status, "other")
    if res.x is None:
        nan = np.full((n, T), np.nan)
        return Plan(status, float("nan"), float("nan"), nan, nan, nan, nan)
    sol = res.x.reshape(n, T, NV)
    gap_val = float(getattr(res, "mip_gap", np.nan) or 0.0)
    return Plan(status, float(res.fun), gap_val, sol[..., X], np.round(sol[..., Y]),
                sol[..., INV], sol[..., BL])


def evaluate(production: np.ndarray, demand: np.ndarray, net0: np.ndarray) -> dict:
    """任一生產計畫（MILP 或實際）在實際需求下的結帳：淨庫存路徑拆成庫存與欠單。"""
    net = net0[:, None] + np.cumsum(production - demand, axis=1)
    inv, bl = np.clip(net, 0, None), np.clip(-net, 0, None)
    return {"mean_inventory_t": float(inv.sum(axis=0).mean()),
            "mean_backlog_t": float(bl.sum(axis=0).mean()),
            "setups": int((production > 1e-6).sum()),
            "end_net": net[:, -1]}


def cost(ev: dict, setup_cost: float, hold: float = 1.0, penalty: float = 10.0, T: int = 1) -> float:
    """用與 MILP 相同的成本函數幫任一計畫計價（平均值 × 週數＝總和）。"""
    return T * (hold * ev["mean_inventory_t"] + penalty * ev["mean_backlog_t"]) + setup_cost * ev["setups"]


def rolling(forecast: np.ndarray, actual: np.ndarray, groups: list, capacity: dict,
            min_batch: np.ndarray, net0: np.ndarray, setup_cost: float, horizon: int = 4,
            safety: np.ndarray | None = None, **kw) -> tuple[np.ndarray, list[str]]:
    """滾動計畫：第 k 週只看得到 forecast[:, k]（該週起點的週預測）與已實現的淨庫存。

    每週以 horizon 週為期求解（未來各週沿用同一週預測），只執行第一週，再用實際需求結帳。
    safety（n × T）給定時，第 k 週用 safety[:, k] 當整個期間的安全庫存軟下限。
    回傳 (n × T 的實際執行產量, 每週求解狀態)。
    """
    n, T = actual.shape
    net = np.asarray(net0, float).copy()
    prod = np.zeros((n, T))
    statuses = []
    for k in range(T):
        d = np.repeat(forecast[:, [k]], horizon, axis=1)
        ss = None if safety is None else np.repeat(safety[:, [k]], horizon, axis=1)
        plan = solve(d, groups, capacity, min_batch, net, setup_cost, safety=ss, **kw)
        statuses.append(plan.status)
        if not np.isfinite(plan.x).all():
            raise RuntimeError(f"第 {k} 週求解失敗：{plan.status}")
        prod[:, k] = np.where(plan.y[:, 0] > 0.5, plan.x[:, 0], 0.0)
        net = net + prod[:, k] - actual[:, k]
    return prod, statuses
