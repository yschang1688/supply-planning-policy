"""KPI 層的守門測試。

KPI 寫錯通常不會報錯：欠貨符號寫反、FIFO 寫成 LIFO、fill rate 沒封頂，程式照跑、
表照出，只是結論錯。所以每組都附一個「邏輯寫反就會紅」的案例，答案都是手算得出來的小序列。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import kpi  # noqa: E402


# ── fill rate ──────────────────────────────────────────────────────────
def test_fill_rate_basic_and_direction():
    # 訂單 100、出貨 90 → 0.9；若參數對調會得到 1.0
    assert kpi.fill_rate([60, 40], [50, 40]) == pytest.approx(0.9)


def test_fill_rate_capped_at_one_when_clearing_old_backlog():
    assert kpi.fill_rate([10], [25]) == 1.0


def test_fill_rate_nan_without_orders():
    assert np.isnan(kpi.fill_rate([0, 0], [5, 0]))


def test_fill_rate_weighted_uses_tons_not_units():
    o = pd.DataFrame({"small": [100.0], "big": [10.0]})
    d = pd.DataFrame({"small": [100.0], "big": [0.0]})
    # 件數口徑 100/110 = 0.909；噸數口徑（small 0.001 噸/件、big 1 噸/件）= 0.1 / 10.1
    w = pd.Series({"small": 0.001, "big": 1.0})
    assert kpi.fill_rate_weighted(o, d, w) == pytest.approx(0.1 / 10.1)


# ── FIFO ──────────────────────────────────────────────────────────────
def test_fifo_serves_oldest_first():
    # 第 0、1 天各下 10；第 1、2 天各出 10。FIFO：兩批都等 1 天。LIFO 會得到 [2, 0]。
    r = kpi.fifo([10, 10, 0], [0, 10, 10])
    assert list(r.lots.days_to_fill) == [1, 1]


def test_fifo_same_day_delivery_counts_as_zero_days():
    r = kpi.fifo([5], [5])
    assert list(r.lots.days_to_fill) == [0]


def test_fifo_partial_then_complete():
    r = kpi.fifo([10, 0, 0], [4, 0, 6])
    assert list(r.lots.days_to_fill) == [2]
    age = r.oldest_open_age
    assert list(age[:2]) == [0, 1] and np.isnan(age[2])   # 交齊當天起不再有未交齊批


def test_fifo_unfilled_lot_is_nan_and_counted_open():
    r = kpi.fifo([10, 10], [10, 3])
    assert r.lots.days_to_fill.iloc[0] == 0
    assert np.isnan(r.lots.days_to_fill.iloc[1])
    assert r.open_at_end == pytest.approx(7)


def test_opening_backlog_delays_new_orders():
    # 第 1 天出貨超前訂單 → 期初至少有 10 的舊單。
    order, delivery = [10, 0], [10, 10]
    b0 = kpi.opening_lower_bound(order, delivery)
    assert b0 == pytest.approx(10)
    no_b0 = kpi.fifo(order, delivery)
    with_b0 = kpi.fifo(order, delivery, opening_backlog=b0)
    assert list(no_b0.lots.days_to_fill) == [0]
    assert no_b0.unmatched_delivery == pytest.approx(10)
    # 舊單先吃掉第 0 天的出貨，新單被推到第 1 天
    assert list(with_b0.lots.days_to_fill) == [1]
    assert with_b0.unmatched_delivery == pytest.approx(0)


def test_opening_backlog_lower_bound_leaves_no_unmatched_delivery():
    rng = np.random.default_rng(0)
    order = rng.integers(0, 20, 60).astype(float)
    delivery = np.roll(order, 2)            # 出貨落後兩天，前兩天其實在送期初舊單
    b0 = kpi.opening_lower_bound(order, delivery)
    assert kpi.fifo(order, delivery, opening_backlog=b0).unmatched_delivery == pytest.approx(0)


def test_otif_curve_monotone_and_counts_unfilled_as_miss():
    lots = pd.DataFrame({"qty": [10, 10, 10], "days_to_fill": [0, 3, np.nan]})
    c = kpi.otif_curve(lots, max_days=4)
    assert c[0]["by_lot"] == pytest.approx(1 / 3)
    assert c[3]["by_lot"] == pytest.approx(2 / 3)
    assert c[4]["by_lot"] == pytest.approx(2 / 3)     # 未交齊永遠不算準時
    vals = [c[d]["by_lot"] for d in range(5)]
    assert vals == sorted(vals)


def test_stockout_flags_by_age():
    r = kpi.fifo([10, 0, 0, 0], [0, 0, 0, 10])
    flags = kpi.stockout_flags(r.oldest_open_age, max_days=1)
    assert list(flags) == [False, False, True, False]


# ── 庫存 ──────────────────────────────────────────────────────────────
def test_stock_level_direction():
    # 流入加、流出減；寫反會得到負庫存
    assert list(kpi.stock_level([10, 0], [0, 5], opening=0)) == [10, 5]


def test_opening_lower_bound_is_minimal():
    inflow, outflow = [0, 5, 0], [5, 0, 3]
    b = kpi.opening_lower_bound(inflow, outflow)
    assert b == pytest.approx(5)
    s = kpi.stock_level(inflow, outflow, b)
    assert s.min() == pytest.approx(0)                # 剛好不為負
    assert kpi.stock_level(inflow, outflow, b - 1).min() < 0


def test_days_of_cover_nan_without_demand():
    stock = np.full(10, 50.0)
    cover = kpi.days_of_cover(stock, np.zeros(10), window=7)
    assert np.isnan(cover).all()
    cover = kpi.days_of_cover(stock, np.full(10, 10.0), window=7)
    assert cover[-1] == pytest.approx(5.0)


# ── ABC × XYZ ─────────────────────────────────────────────────────────
def test_abc_boundaries():
    v = pd.Series({"a": 70, "b": 20, "c": 6, "d": 4})
    assert kpi.abc_class(v).to_dict() == {"a": "A", "b": "A", "c": "B", "d": "C"}


def test_abc_item_starting_exactly_at_boundary_is_next_class():
    # 前面已累計剛好 80% 的品項不屬 A（否則 A 類會比 80% 多吃一個品項）
    v = pd.Series({"a": 80, "b": 15, "c": 5})
    assert kpi.abc_class(v).to_dict() == {"a": "A", "b": "B", "c": "C"}


def test_xyz_uses_full_weeks_and_cv():
    idx = pd.date_range("2023-01-01", periods=28, freq="D")   # 週日起 → 4 個完整的週日到週六
    steady = np.tile([1, 1, 1, 1, 0, 1, 1], 4).astype(float)  # 每週相同 → CV 0
    lumpy = np.zeros(28)
    lumpy[[3, 24]] = 50                                        # 只有兩週有量 → CV 1
    df = pd.DataFrame({"steady": steady, "lumpy": lumpy}, index=idx)
    out = kpi.xyz_class(df)
    assert out["steady"] == "X"
    assert out["lumpy"] == "Z"


def test_segment_marks_dormant():
    idx = pd.date_range("2023-01-01", periods=14, freq="D")
    df = pd.DataFrame({"x": np.ones(14), "dead": np.zeros(14)}, index=idx)
    seg = kpi.segment(df, dormant=["dead"])
    assert seg.loc["dead", "segment"] == "D"
    assert seg.loc["x", "segment"] == "AX"


# ── 預測誤差 ──────────────────────────────────────────────────────────
def test_wape_and_bias_sign():
    a = [10, 10]
    assert kpi.wape(a, [12, 8]) == pytest.approx(0.2)
    assert kpi.bias(a, [12, 12]) > 0      # 高估為正
    assert kpi.bias(a, [8, 8]) < 0


def test_wape_nan_when_no_actual_demand():
    # 回 0 會把沒有需求的 SKU 算成完美預測
    assert np.isnan(kpi.wape([0, 0], [1, 0]))
    assert np.isnan(kpi.bias([0, 0], [1, 0]))
