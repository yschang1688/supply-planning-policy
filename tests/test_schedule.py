"""M4 產能排程 MILP 的守門測試。

MILP 寫錯的典型樣子：平衡式少一項、最小批量的不等號反了、產能限制只套到第一週。
求解器照樣回 optimal，只是解的是另一個問題。所以這裡逐條驗限制、驗參數單調，並驗無解判定。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import schedule as M  # noqa: E402

GROUPS = ["A", "A", "A", "B", "B"]


def _inst(seed=0, T=6):
    rng = np.random.default_rng(seed)
    d = rng.uniform(4, 20, (5, T))
    cap = {"A": 45.0, "B": 35.0}
    mb = np.array([10, 12, 8, 9, 15], float)
    return d, cap, mb, np.zeros(5)


# ── 無解判定 ──────────────────────────────────────────────────────────
def test_infeasible_without_backlog_when_capacity_short():
    d = np.full((2, 3), 50.0)
    cap = {"A": 60.0}                       # 每週需求 100、產能 60
    p = M.solve(d, ["A", "A"], cap, np.ones(2), np.zeros(2), 1.0, allow_backlog=False)
    assert p.status == "infeasible"
    assert np.isnan(p.x).all()


def test_backlog_absorbs_shortage_when_allowed():
    d = np.full((2, 3), 50.0)
    cap = {"A": 60.0}
    p = M.solve(d, ["A", "A"], cap, np.ones(2), np.zeros(2), 1.0, allow_backlog=True)
    assert p.status == "optimal"
    assert p.backlog[:, -1].sum() >= 3 * (100 - 60) - 1e-6


def test_non_finite_input_is_rejected():
    d, cap, mb, net0 = _inst()
    mb[2] = np.nan
    with pytest.raises(ValueError, match="min_batch"):
        M.solve(d, GROUPS, cap, mb, net0, 5.0)


# ── 限制逐條成立 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("s", [0.0, 20.0, 200.0])
def test_constraints_hold(seed, s):
    d, cap, mb, net0 = _inst(seed)
    p = M.solve(d, GROUPS, cap, mb, net0, s, gap=0)
    assert p.status == "optimal"
    for g, c in cap.items():
        idx = [i for i, x in enumerate(GROUPS) if x == g]
        assert (p.x[idx].sum(axis=0) <= c + 1e-6).all()            # 每週每群產能
    on = p.y > 0.5
    assert (p.x[on] >= np.repeat(mb[:, None], d.shape[1], 1)[on] - 1e-6).all()   # 最小批量
    assert (p.x[~on] <= 1e-6).all()                                 # 沒開線就不能生產
    net = net0[:, None] + np.cumsum(p.x - d, axis=1)
    np.testing.assert_allclose(net, p.inv - p.backlog, atol=1e-5)   # 平衡式


def test_terminal_condition_is_enforced():
    d, cap, mb, net0 = _inst()
    term = np.full(5, 10.0)
    p = M.solve(d, GROUPS, cap, mb, net0, 50.0, terminal=term, gap=0)
    assert ((p.inv[:, -1] - p.backlog[:, -1]) >= term - 1e-6).all()


def test_safety_stock_is_held_when_cheap():
    d, cap, mb, net0 = _inst()
    ss = np.full(d.shape, 5.0)
    p = M.solve(d, GROUPS, cap, mb, net0, 0.0, safety=ss, safety_penalty=50.0, gap=0)
    assert (p.inv >= 5.0 - 1e-6).all()
    q = M.solve(d, GROUPS, cap, mb, net0, 0.0, gap=0)
    assert p.inv.mean() > q.inv.mean()


# ── 參數單調 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", range(3))
def test_relaxing_constraints_never_worsens_objective(seed):
    d, cap, mb, net0 = _inst(seed)
    base = M.solve(d, GROUPS, cap, mb, net0, 20.0, gap=0).objective
    looser_batch = M.solve(d, GROUPS, cap, mb * 0.5, net0, 20.0, gap=0).objective
    more_cap = M.solve(d, GROUPS, {g: c * 1.5 for g, c in cap.items()}, mb, net0, 20.0, gap=0).objective
    assert looser_batch <= base + 1e-6
    assert more_cap <= base + 1e-6


@pytest.mark.parametrize("seed", range(3))
def test_higher_setup_cost_means_fewer_setups(seed):
    d, cap, mb, net0 = _inst(seed, T=8)
    prev_setups, prev_other = 10 ** 9, -1.0
    for s in (0.0, 5.0, 20.0, 80.0, 300.0):
        p = M.solve(d, GROUPS, cap, mb, net0, s, gap=0)
        other = p.objective - s * p.setups                         # 庫存＋欠單成本
        assert p.setups <= prev_setups
        assert other >= prev_other - 1e-6
        prev_setups, prev_other = p.setups, other


# ── 結帳 ──────────────────────────────────────────────────────────────
def test_evaluate_splits_net_into_inventory_and_backlog():
    prod = np.array([[10.0, 0.0, 0.0]])
    dem = np.array([[4.0, 4.0, 4.0]])
    ev = M.evaluate(prod, dem, np.array([0.0]))
    # 淨庫存 6、2、−2 → 庫存 6、2、0；欠單 0、0、2
    assert ev["mean_inventory_t"] == pytest.approx(8 / 3)
    assert ev["mean_backlog_t"] == pytest.approx(2 / 3)
    assert ev["setups"] == 1
    assert ev["end_net"][0] == pytest.approx(-2)


# ── 滾動計畫不偷看未來 ────────────────────────────────────────────────
def test_rolling_decisions_ignore_future_actuals():
    d, cap, mb, net0 = _inst(T=6)
    fc = d * 1.1
    a, _ = M.rolling(fc, d, GROUPS, cap, mb, net0, 10.0, gap=0)
    tampered = d.copy()
    tampered[:, 3:] *= 5
    b, _ = M.rolling(fc, tampered, GROUPS, cap, mb, net0, 10.0, gap=0)
    # 第 3 週的決策只用到第 0～2 週的實際需求
    np.testing.assert_allclose(a[:, :4], b[:, :4])
    assert not np.allclose(a, b)
