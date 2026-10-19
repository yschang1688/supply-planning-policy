"""已落盤結果的宣稱守門：results/*.json 是 README 數字的出處，這裡確認它們沒有悄悄背離設計。

不重跑管線（CI 不下載資料），只檢查落盤結果彼此一致、方向合理。
"""
import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "results"


@pytest.fixture(scope="module")
def q():
    return json.loads((RESULTS / "m0_quality.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def m1():
    return json.loads((RESULTS / "m1_kpi.json").read_text(encoding="utf-8"))


def test_data_shape_matches_design(q):
    assert q["dates"] == {"start": "2023-01-01", "end": "2023-08-09", "days": 221}
    assert q["n_skus"] == 41
    assert "POV005L04P/factory_issue" in q["ton_ratio"]["weight_column_missing"]


def test_scope_excludes_dormant(q, m1):
    assert m1["scope"]["dormant_skus"] == q["dormant_skus"]["count"]
    assert m1["scope"]["active_skus"] + m1["scope"]["dormant_skus"] == q["n_skus"]


def test_fill_rate_is_a_rate(m1):
    assert 0 < m1["fill_rate_ton_weighted"] <= 1


@pytest.mark.parametrize("variant", ["fifo_opening_backlog_zero", "fifo_opening_backlog_lower_bound"])
def test_otif_curve_monotone_and_stockout_decreasing(m1, variant):
    v = m1[variant]
    otif = [v["otif_proxy_by_d"][str(d)]["by_lot"] for d in range(8)]
    stock = [v["stockout_rate_by_d"][str(d)] for d in range(8)]
    assert otif == sorted(otif)
    assert stock == sorted(stock, reverse=True)


def test_opening_backlog_never_makes_service_look_better(m1):
    """期初欠貨只會把新單往後推：下限版的 OTIF 不得高於零版，缺貨率不得低於零版。"""
    a, b = m1["fifo_opening_backlog_zero"], m1["fifo_opening_backlog_lower_bound"]
    for d in range(8):
        assert b["otif_proxy_by_d"][str(d)]["by_lot"] <= a["otif_proxy_by_d"][str(d)]["by_lot"]
        assert b["stockout_rate_by_d"][str(d)] >= a["stockout_rate_by_d"][str(d)]
    assert b["unmatched_delivery_tons"] == 0


def test_segment_counts_cover_all_skus(q, m1):
    assert sum(m1["segments"].values()) == q["n_skus"]
    assert m1["segments"]["D"] == q["dormant_skus"]["count"]
