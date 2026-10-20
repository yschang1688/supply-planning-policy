"""M3 庫存模擬的不變量測試。

模擬器寫錯的典型樣子：到貨日少算一天、欠單沒補、安全庫存偷看了下一週的誤差。
這些都不會報錯，只會讓某個政策看起來特別好。所以這裡測的是不變量：守恆、單調、不偷看未來。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import simulate as S  # noqa: E402


def _case(seed=0, days=63, noisy=True):
    rng = np.random.default_rng(seed)
    week = np.array([12, 9, 8, 14, 0, 3, 16], float)
    demand = np.r_[0, np.tile(week, days // 7 + 1)[:days]]
    if noisy:
        demand = demand * rng.uniform(0.3, 1.8, len(demand))
    fc = {o: week.copy() for o in range(0, days, 7)}
    ss = {o: 0.0 for o in fc}
    return demand, fc, ss


# ── 完美預測 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("policy", ["RS", "sS"])
@pytest.mark.parametrize("lead", [1, 2, 5])
def test_perfect_forecast_never_stocks_out(policy, lead):
    demand, fc, ss = _case(noisy=False)
    r = S.simulate(demand, fc, ss, policy=policy, lead=lead)
    assert r.fill_rate == pytest.approx(1.0)
    assert r.end_backlog == pytest.approx(0)
    assert not any(r.cycle_stockout)


# ── 守恆 ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("policy", ["RS", "sS"])
@pytest.mark.parametrize("seed", range(5))
def test_inventory_and_demand_conservation(policy, seed):
    demand, fc, ss = _case(seed)
    rng = np.random.default_rng(seed + 100)
    al = rng.integers(1, 6, len(demand))
    r = S.simulate(demand, fc, {o: 5.0 for o in ss}, policy=policy, lead=2, actual_lead=al)
    assert r.initial_on_hand + r.received - r.shipped == pytest.approx(r.end_on_hand)
    assert r.shipped + r.end_backlog == pytest.approx(r.demand)
    assert (r.on_hand >= -1e-9).all()


def test_backlog_counts_in_inventory_position_and_is_served_first():
    # S = 1 × (7+1) = 8。第 1 天需求 30 → 出 8、欠 22。
    # 第 7 天檢視：庫存位置 = 0 + 0 − 22 = −22 → 補 30，第 8 天到貨。
    # 第 8 天：先補欠單 22，再出當天的 5 → 在手 3、欠單 0。
    demand = np.r_[0, 30, 0, 0, 0, 0, 0, 0, 5]
    fc = {0: np.full(7, 1.0), 7: np.full(7, 1.0)}
    r = S.simulate(demand, fc, {0: 0.0, 7: 0.0}, policy="RS", lead=1)
    assert r.unmet_today[0] == pytest.approx(22)
    assert r.on_hand[7] == pytest.approx(3)
    assert r.unmet_today[7] == 0
    assert r.end_backlog == pytest.approx(0)


# ── 單調 ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("policy", ["RS", "sS"])
@pytest.mark.parametrize("seed", range(5))
def test_more_safety_stock_never_hurts_service(policy, seed):
    demand, fc, _ = _case(seed)
    prev_fill = prev_oh = -1
    for level in (0, 5, 20, 60):
        r = S.simulate(demand, fc, {o: float(level) for o in fc}, policy=policy, lead=2)
        assert r.fill_rate >= prev_fill - 1e-9
        assert r.on_hand.mean() >= prev_oh - 1e-9
        prev_fill, prev_oh = r.fill_rate, r.on_hand.mean()


@pytest.mark.parametrize("policy", ["RS", "sS"])
@pytest.mark.parametrize("seed", range(5))
def test_longer_actual_lead_never_helps(policy, seed):
    demand, fc, _ = _case(seed)
    ss = {o: 10.0 for o in fc}
    prev = 2.0
    for L in (2, 3, 5, 8):
        r = S.simulate(demand, fc, ss, policy=policy, lead=2, actual_lead=np.full(len(demand), L))
        assert r.fill_rate <= prev + 1e-9
        prev = r.fill_rate


def test_arrival_lands_exactly_lead_days_later():
    # 預測每天 2、SS 0、lead 3 → S = 2 × (7+3) = 20。第 7 天日終在手 6，補 14，第 10 天到貨。
    demand = np.r_[0, np.full(10, 2.0)]
    fc = {0: np.full(7, 2.0), 7: np.full(7, 2.0)}
    r = S.simulate(demand, fc, {0: 0.0, 7: 0.0}, policy="RS", lead=3)
    assert r.initial_on_hand == pytest.approx(20)
    assert r.on_hand[8] == pytest.approx(2)      # 第 9 天：還沒到
    assert r.on_hand[9] == pytest.approx(14)     # 第 10 天：到貨 14、出 2


# ── 安全係數 ──────────────────────────────────────────────────────────
def test_safety_factor_kinds():
    r = np.r_[-1.0, 1.0] .repeat(50)            # 平均 0、標準差 ≈1
    assert S.safety_factor(r, 0.5, "normal_classic") == pytest.approx(0)
    shifted = r + 0.5
    assert S.safety_factor(shifted, 0.84, "normal") == pytest.approx(
        0.5 + S.safety_factor(shifted, 0.84, "normal_classic"), rel=1e-6)
    # 分位數取「higher」：寧可多備一格，不在樣本之間內插出比任何觀測都小的值
    assert S.safety_factor(np.arange(1, 101) / 100, 0.95, "empirical") == pytest.approx(0.96)


def test_safety_factor_monotone_in_alpha_and_never_negative():
    r = np.random.default_rng(0).lognormal(0, 1, 200) - 1.5
    for kind in ("normal_classic", "normal", "empirical"):
        q = [S.safety_factor(r, a, kind) for a in (0.8, 0.9, 0.95, 0.99)]
        assert q == sorted(q)
    assert S.safety_factor(np.full(10, -3.0) + np.linspace(0, 0.1, 10), 0.9, "normal") == 0


def test_skewed_errors_make_empirical_exceed_normal():
    # 右偏：常態近似低估高分位數——H2 的機制本身
    r = np.random.default_rng(1).lognormal(0, 1, 5000)
    assert S.safety_factor(r, 0.99, "empirical") > S.safety_factor(r, 0.99, "normal")


def test_safety_factor_needs_data():
    with pytest.raises(ValueError):
        S.safety_factor([1.0], 0.9, "normal")


# ── 不偷看未來 ────────────────────────────────────────────────────────
def test_past_pool_excludes_cutoff_and_later():
    idx = pd.date_range("2023-01-07", periods=6, freq="7D")
    resid = pd.DataFrame(np.arange(12, dtype=float).reshape(6, 2), index=idx, columns=["a", "b"])
    cutoff = idx[3]
    before = S.past_pool(resid, cutoff, ["a", "b"])
    tampered = resid.copy()
    tampered.loc[cutoff:] = 999.0
    np.testing.assert_array_equal(before, S.past_pool(tampered, cutoff, ["a", "b"]))
    assert len(before) == 3 * 2
    assert len(S.past_pool(resid, cutoff, ["a"])) == 3
