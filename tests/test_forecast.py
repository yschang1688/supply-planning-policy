"""M2 需求預測的守門測試。

預測最常見的錯誤不是模型不準，是**偷看未來**：特徵多讀了一天、訓練列的目標日跨過起點。
這種錯不會報錯，只會讓回測好看。所以第一組測試直接竄改起點之後的值，預測必須一個位元都不變。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import forecast as F  # noqa: E402


def _panel(n=84, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    base = np.tile([5, 4, 3, 6, 0, 1, 7], n // 7 + 1)[:n]
    Y = base[:, None] * rng.uniform(0.5, 1.5, (n, k)) + rng.poisson(1, (n, k))
    return pd.DataFrame(Y, index=idx, columns=[f"s{i}" for i in range(k)])


# ── 一、不偷看未來 ────────────────────────────────────────────────────
@pytest.mark.parametrize("method", ["seasonal_naive", "weekday_mean", "ma28", "ets",
                                    "croston", "tsb", "lgbm", "lgbm_calendar"])
def test_future_values_cannot_change_forecast(method):
    order = _panel()
    groups = pd.Series({c: "G" for c in order.columns})
    o = 60
    a = F.backtest(order, groups, [o], methods=(method,))
    tampered = order.copy()
    tampered.iloc[o + 1:] = tampered.iloc[o + 1:] * 100 + 999
    b = F.backtest(tampered, groups, [o], methods=(method,))
    np.testing.assert_array_equal(a[method].to_numpy(), b[method].to_numpy())
    assert not np.array_equal(a["actual"].to_numpy(), b["actual"].to_numpy())  # 竄改確實生效


def test_lgbm_training_rows_never_cross_origin():
    order = _panel()
    Y = order.to_numpy()
    cal = F.calendar(order.index)
    origin = 50
    rows = pd.concat([F._rows(Y, o, order.index, list(order.columns), ["G"] * 3, cal,
                              last_known=origin) for o in range(27, origin)])
    assert rows.t.max() == origin
    assert (rows.t > rows.origin).all()


def test_origins_leave_full_horizon():
    idx = F.origins(221)
    assert idx[0] == 111 and len(idx) == 15
    assert idx[-1] + F.H <= 220
    tight = F.origins(120, first=100, step=1)     # 逐日起點時邊界才會被踩到
    assert tight[-1] + F.H == 119


# ── 二、基準方法 ──────────────────────────────────────────────────────
def test_fixed_weekly_pattern_is_reproduced_exactly():
    week = np.array([5, 4, 3, 6, 0, 1, 7], float)
    hist = np.tile(week, 8)                       # 56 天，下一天是 week[0]
    np.testing.assert_allclose(F.seasonal_naive(hist), week)
    np.testing.assert_allclose(F.weekday_mean(hist), week)
    assert not np.allclose(F.moving_average(hist), week)


def test_weekday_mean_reads_same_weekday_not_yesterday():
    hist = np.zeros(28)
    hist[-7] = 10                                 # 上週同一天（h=1 的同星期）
    hist[-1] = 99                                 # 昨天
    assert F.weekday_mean(hist, h=1)[0] == pytest.approx(10 / 4)


# ── 三、間歇需求 ──────────────────────────────────────────────────────
def test_croston_and_tsb_recover_constant_rate():
    hist = np.full(60, 3.0)
    assert F.croston_rate(hist) == pytest.approx(3.0)
    assert F.tsb_rate(hist) == pytest.approx(3.0, rel=1e-3)


def test_croston_rate_for_regular_intermittent_demand():
    hist = np.tile([0, 0, 6.0], 30)               # 每 3 天 6 → 每天 2
    assert F.croston_rate(hist) == pytest.approx(2.0, rel=0.05)


def test_tsb_decays_after_demand_stops_croston_does_not():
    hist = np.r_[np.full(40, 5.0), np.zeros(30)]
    c_before, c_after = F.croston_rate(hist[:40]), F.croston_rate(hist)
    t_before, t_after = F.tsb_rate(hist[:40]), F.tsb_rate(hist)
    assert c_after == pytest.approx(c_before)
    assert t_after < 0.1 * t_before


def test_intermittent_profile_restores_weekday_shape():
    week = np.array([5, 4, 3, 6, 0, 1, 7], float)
    hist = np.tile(week, 8)
    fc = F.intermittent(hist, start_dow=6, kind="croston")   # 2023-01-01 是週日
    assert fc[4] == pytest.approx(0)              # 第 5 天對應 week[4] = 0
    assert fc.sum() == pytest.approx(F.croston_rate(hist) * 7)


# ── 四、行事曆 ────────────────────────────────────────────────────────
def test_calendar_flags_eid_and_counts_days_since():
    cal = F.calendar(pd.date_range("2023-04-19", "2023-05-06"))
    assert cal.loc["2023-04-22", "eid_block"] == 1
    assert cal.loc["2023-04-19", "eid_block"] == 0
    assert cal.loc["2023-04-19", "gov_holiday"] == 1
    assert cal.loc["2023-04-20", "days_since_eid"] == 0
    assert cal.loc["2023-04-27", "days_since_eid"] == 7
    assert cal.loc["2023-05-06", "days_since_eid"] == 15   # 超過 14 天封頂


def test_backtest_shape_and_actuals():
    order = _panel()
    bt = F.backtest(order, pd.Series({c: "G" for c in order.columns}), [40, 47],
                    methods=("weekday_mean",))
    assert len(bt) == 2 * F.H * order.shape[1]
    row = bt.iloc[0]
    assert row.actual == order.loc[row.date, row.sku]
    assert (bt.date > bt.origin).all()
