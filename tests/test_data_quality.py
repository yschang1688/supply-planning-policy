"""資料契約與品質檢核的守門測試：每條檢核都要在植入對應的髒資料時報錯。

全部用人工構造的小表，不需要下載資料，CI 離線可跑。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data  # noqa: E402
import quality  # noqa: E402


def _wide(values, start="2023-01-01", skus=("A", "B")):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(values, index=idx, columns=list(skus), dtype=float)


def _flows(n=14, tpu=(0.5, 2.0)):
    rng = np.random.default_rng(1)
    unit = {f: _wide(rng.integers(0, 10, (n, 2))) for f in data.FLOWS}
    ton = {f: unit[f] * np.array(tpu) for f in data.FLOWS}
    flows = data.wide_to_long(unit, ton).rename(columns={"qty_ton": "qty_ton_reported"})
    flows["qty_ton"] = flows.qty_ton_reported
    return flows


# ── sha256 ────────────────────────────────────────────────────────────
def test_checksum_mismatch_names_the_file(tmp_path):
    (tmp_path / "x.csv").write_text("Date,A\n2023-01-01,1\n")
    good = data.sha256_of(tmp_path / "x.csv")
    data.verify(tmp_path, {"x.csv": ("", good)})                 # 相符就過
    with pytest.raises(data.ChecksumError, match="x.csv"):
        data.verify(tmp_path, {"x.csv": ("", "0" * 64)})


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="y.csv"):
        data.verify(tmp_path, {"y.csv": ("", "0" * 64)})


def test_pinned_checksums_are_full_length():
    assert all(len(h) == 64 for _, h in data.FILES.values())
    assert len(data.UPSTREAM_COMMIT) == 40


# ── 寬表轉長表 ────────────────────────────────────────────────────────
def test_wide_to_long_shape_and_alignment():
    f = _flows(n=10)
    assert len(f) == 10 * 2 * 4
    assert set(f.flow) == set(data.FLOWS)


def test_wide_to_long_rejects_misaligned_units():
    unit = {fl: _wide(np.ones((5, 2))) for fl in data.FLOWS}
    ton = {fl: _wide(np.ones((5, 2))) for fl in data.FLOWS}
    ton["order"] = _wide(np.ones((5, 2)), start="2023-01-02")
    with pytest.raises(ValueError, match="order"):
        data.wide_to_long(unit, ton)


# ── 硬錯誤 ────────────────────────────────────────────────────────────
def test_date_gap_is_hard_error():
    w = _wide(np.ones((5, 2)))
    quality.check_dates(w)
    with pytest.raises(quality.HardError):
        quality.check_dates(w.drop(w.index[2]))


def test_negative_is_hard_error():
    f = _flows()
    quality.check_non_negative(f)
    f.loc[3, "qty_unit"] = -1
    with pytest.raises(quality.HardError):
        quality.check_non_negative(f)


def test_inconsistent_ton_ratio_is_hard_error():
    f = _flows()
    quality.check_ton_ratio(f)
    i = f.index[(f.sku == "A") & (f.qty_unit > 0)][0]
    f.loc[i, "qty_ton_reported"] *= 1.5
    with pytest.raises(quality.HardError, match="A"):
        quality.check_ton_ratio(f)


def test_ton_without_units_is_hard_error():
    f = _flows()
    i = f.index[f.qty_unit == 0][0]
    f.loc[i, "qty_ton_reported"] = 3.0
    with pytest.raises(quality.HardError):
        quality.check_ton_ratio(f)


def test_whole_missing_weight_column_is_reported_not_fatal():
    f = _flows()
    m = (f.sku == "B") & (f.flow == "factory_issue")
    f.loc[m, "qty_ton_reported"] = 0.0
    out = quality.check_ton_ratio(f)
    assert out["weight_column_missing"] == ["B/factory_issue"]


def test_partially_missing_weight_is_hard_error():
    f = _flows()
    m = f.index[(f.sku == "B") & (f.flow == "factory_issue") & (f.qty_unit > 0)][:2]
    f.loc[m, "qty_ton_reported"] = 0.0
    with pytest.raises(quality.HardError, match="B/factory_issue"):
        quality.check_ton_ratio(f)


# ── 觀察項 ────────────────────────────────────────────────────────────
def test_dormant_threshold():
    w = _wide(np.c_[np.r_[np.zeros(99), 1], np.ones(100)])
    assert quality.dormant_skus(w) == ["A"]


def test_weekday_profile_detects_dead_day():
    w = _wide(np.ones((14, 2)))
    w[w.index.dayofweek == 4] = 0              # 週五全 0
    prof = quality.weekday_profile(w)
    assert prof["Fri"] == 0 and prof["Mon"] == 2


def test_opening_lower_bound_per_sku():
    inflow = _wide([[0, 5], [0, 0]])
    outflow = _wide([[3, 0], [0, 2]])
    b = quality.opening_lower_bound(inflow, outflow)
    assert b["A"] == 3 and b["B"] == 0
