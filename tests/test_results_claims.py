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


# ── M2：預測的兩個結論 ──────────────────────────────────────────────────
@pytest.fixture(scope="module")
def m2():
    return json.loads((RESULTS / "m2_forecast.json").read_text(encoding="utf-8"))


def test_m2_backtest_design(m2):
    d = m2["design"]
    assert d["n_origins"] == 15 and d["horizon_days"] == 7 and d["skus"] == 32
    assert d["holiday_origins"] == ["2023-04-22", "2023-06-24"]


def test_m2_ignoring_weekday_is_clearly_worse(m2):
    """結論一：不分星期的移動平均在每個起點都輸給星期平均——星期效應是真的。"""
    p = m2["paired_vs_reference"]["all_origins"]["ma28"]
    assert p["a_wins"] == 0 and p["wilcoxon_p"] < 0.01


def test_m2_model_choice_within_noise_outside_holidays(m2):
    """結論二：排除節日週後，沒有任何方法顯著贏過星期平均（README 依此寫「換模型在雜訊內」）。"""
    for name, p in m2["paired_vs_reference"]["excluding_holiday_origins"].items():
        if name == "ma28":
            continue
        assert not (p["median_diff"] < 0 and p["wilcoxon_p"] < 0.05), name


def test_m2_l1_objective_underforecasts(m2):
    """L1 目標預測的是中位數，需求右偏時會系統性低估——M3 的安全庫存必須看到這個 bias。"""
    assert m2["overall"]["lgbm"]["bias"] < 0
    assert m2["overall"]["lgbm_calendar"]["bias"] < 0


# ── M3：庫存政策 ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def m3():
    return json.loads((RESULTS / "m3_policy.json").read_text(encoding="utf-8"))


def test_m3_design(m3):
    d = m3["design"]
    assert d["warmup_origins"] == 4 and len(d["measured_origins"]) == 11
    assert d["skus"] == 32 and d["class_sizes"] == {"X": 20, "YZ": 12}


def test_m3_base_curve_monotone_in_target(m3):
    """同一需求路徑上，目標越高，庫存不會變少、fill rate 不會變差。"""
    c = m3["base_curve"]
    alphas = sorted(c, key=float)
    cover = [c[a]["days_of_cover"] for a in alphas]
    fill = [c[a]["fill_rate"] for a in alphas]
    assert cover == sorted(cover)
    assert fill == sorted(fill)


def test_m3_h1_and_h2_flags_match_their_numbers(m3):
    h1 = m3["h1"]
    lo = h1["target_minus_method_days"]["ci95"][0]
    assert h1["supported"] == (h1["bootstrap_valid_share"] >= 0.9 and lo > 0)
    covers = h1["method_effect_cover_at_equal_fill"]["cover_by_method"]
    assert all(v is not None for v in covers.values())     # 每個方法都內插得到
    sf = m3["h2"]["mean_shortfall_alpha_ge_0.90"]
    expect = sf["normal"]["YZ"] < sf["normal"]["X"] - 0.02 and sf["empirical"]["YZ"] > sf["normal"]["YZ"]
    assert m3["h2"]["supported"] == expect


def test_m3_readme_verdicts(m3):
    """README 的兩句判定：H1 成立、H2 不成立。結果翻轉時這裡先紅，README 不能照舊。"""
    assert m3["h1"]["supported"] is True
    assert m3["h2"]["supported"] is False
    eq = m3["h1"]["method_effect_cover_at_equal_fill"]
    assert m3["h1"]["target_effect_0.80_to_0.99"]["days_of_cover"] > 3 * eq["range_days"]


# ── M4：產能排程 ────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def m4():
    return json.loads((RESULTS / "m4_schedule.json").read_text(encoding="utf-8"))


def test_m4_all_solves_optimal(m4):
    assert m4["all_optimal"]
    assert m4["rolling"]["no_safety_stock"]["statuses"] == ["optimal"]
    assert m4["rolling"]["with_safety_stock"]["statuses"] == ["optimal"]
    assert m4["design"]["excluded_never_produced"] == ["SO0002L09P"]


def test_m4_frontier_is_a_tradeoff(m4):
    f = sorted(m4["frontier"], key=lambda r: r["setup_cost"])
    setups = [r["setups"] for r in f]
    assert setups == sorted(setups, reverse=True)
    # 單調的是「庫存＋欠單成本」，不是庫存本身；MIP gap 0.5% 內的抖動容許
    other = [r["mean_inventory_t"] + 10 * r["mean_backlog_t"] for r in f]
    tol = [0.005 * r["objective"] / 11 for r in f]
    assert all(b >= a - t for a, b, t in zip(other, other[1:], tol[1:]))
    assert other[-1] > other[0] * 1.3


def test_m4_readme_findings(m4):
    """README 的三句：完全資訊可少換線、沒安全庫存的滾動計畫會大量欠單、
    加了安全庫存仍貴過實際計畫（實際的逐日小批量節奏在這份資料上很難打敗）。"""
    assert m4["perfect_info_same_setup_cost"]["setups"] < m4["actual_plan"]["setups"]
    r = m4["rolling"]
    assert r["no_safety_stock"]["mean_backlog_t"] > 10 * r["with_safety_stock"]["mean_backlog_t"]
    assert r["with_safety_stock"]["cost"] > m4["actual_cost_same_setup_cost"]
