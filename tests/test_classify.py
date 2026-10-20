"""M5 欠貨風險分類的守門測試：特徵不偷看未來、標籤定義、時間切分的清空期。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import classify as C  # noqa: E402


def _flows(n=70, k=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    cols = [f"s{i}" for i in range(k)]
    order = pd.DataFrame(rng.poisson(10, (n, k)).astype(float), index=idx, columns=cols)
    delivery = order.shift(2).fillna(0) * rng.uniform(0.6, 1.2, (n, k))
    production = order * rng.uniform(0.5, 1.5, (n, k)) * (rng.random((n, k)) < 0.7)
    issue = production.shift(1).fillna(0)
    tpu = pd.Series(0.01, index=cols)
    groups = pd.Series("G", index=cols)
    return order, delivery, production, issue, tpu, groups


# ── 特徵不偷看未來 ────────────────────────────────────────────────────
def test_features_ignore_future_flows():
    o, d, p, i, tpu, g = _flows()
    cut = 40
    a = C.features(o, d, p, i, tpu, g)
    t2 = [x.copy() for x in (o, d, p, i)]
    for x in t2:
        x.iloc[cut + 1:] = x.iloc[cut + 1:] * 7 + 3
    b = C.features(*t2, tpu, g)
    cols = [c for c in C.FEATURES if c != "eid_next7"]          # 假期是事前公告的行事曆
    left = a[a.t <= cut][cols].to_numpy()
    right = b[b.t <= cut][cols].to_numpy()
    np.testing.assert_allclose(left, right)
    assert not np.allclose(a[a.t > cut][cols].to_numpy(), b[b.t > cut][cols].to_numpy())


def test_eid_next7_is_calendar_only():
    o, d, p, i, tpu, g = _flows(n=130)
    f = C.features(o, d, p, i, tpu, g)
    row = f[(f.sku == "s0") & (f.date == "2023-04-14")]          # 04-20 開齋節連假在 7 天內
    assert row.eid_next7.iloc[0] == 1
    row = f[(f.sku == "s0") & (f.date == "2023-04-01")]
    assert row.eid_next7.iloc[0] == 0


# ── 標籤 ──────────────────────────────────────────────────────────────
def test_onset_and_any7_labels():
    idx = pd.date_range("2023-01-01", periods=20, freq="D")
    stock = pd.DataFrame({"a": np.zeros(20, bool)}, index=idx)
    stock.iloc[10:13, 0] = True                                 # 第 10～12 天缺貨
    now, any7 = C.labels(stock)
    assert any7.iloc[3, 0] == 1 and any7.iloc[2, 0] == 0         # 第 3 天看得到第 10 天，第 2 天看不到
    assert any7.iloc[9, 0] == 1 and now.iloc[9, 0] == 0          # 第 9 天：今天沒缺、明天開始缺 → onset 1
    assert now.iloc[10, 0] == 1                                  # 第 10 天已在缺貨 → 不進 onset 模型
    assert any7.iloc[12, 0] == 0                                 # 第 13～19 天都沒缺
    assert any7.iloc[-7:, 0].isna().all()                        # 看不到完整 7 天的列


def test_panel_onset_excludes_rows_already_in_stockout():
    o, d, p, i, tpu, g = _flows()
    P = C.panel(o, d, p, i, tpu, g)
    assert P.loc[P.now == 1, "onset"].isna().all()
    ok = P.now == 0
    np.testing.assert_array_equal(P.loc[ok, "onset"].to_numpy(), P.loc[ok, "any7"].to_numpy())


# ── 時間切分 ──────────────────────────────────────────────────────────
def test_time_split_purges_label_window():
    t = pd.Series(np.arange(221))
    train, test = C.time_split(t)
    assert t[train].max() + C.HORIZON < t[test].min()            # 訓練標籤期早於第一個測試日
    assert t[test].max() + C.HORIZON <= 220                      # 測試標籤看得到完整 7 天
    assert not (train & test).any()


def test_time_split_rejects_short_purge():
    with pytest.raises(ValueError):
        C.time_split(pd.Series(np.arange(221)), purge=3)


def test_inner_threshold_split_also_purged():
    assert C.INNER_FIT_END + C.HORIZON <= C.INNER_VAL_START
