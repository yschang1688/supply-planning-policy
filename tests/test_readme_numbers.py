"""README 的每個數字都要能在 results/*.json 找到：從落盤結果重新排版，逐一確認 README 裡有那個字串。

結果重跑後數字變了、README 沒跟著改，這裡就紅。
"""
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _j(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def pct(x):
    return f"{x * 100:.0f}%"


def expected() -> dict[str, str]:
    m1, m2, m3, m4, m5 = (_j(f) for f in ("m1_kpi.json", "m2_forecast.json", "m3_policy.json",
                                           "m4_schedule.json", "m5_classify.json"))
    fifo = m1["fifo_opening_backlog_lower_bound"]
    h1 = m3["h1"]
    t, m = h1["target_effect_0.80_to_0.99"], h1["method_effect_cover_at_equal_fill"]
    ex = m2["excluding_holiday_windows"]
    pe = m2["paired_vs_reference"]["excluding_holiday_origins"]["lgbm_calendar"]
    pa = m2["paired_vs_reference"]["all_origins"]["ma28"]
    sf = m3["h2"]["mean_shortfall_alpha_ge_0.90"]["normal"]
    roll = m4["rolling"]
    on = m5["onset"]
    boot = on["bootstrap_pr_auc_vs_best_rule"]["logistic"]
    mc = m3["monte_carlo_lead_noise"]["0.95"]["fill_rate"]
    base = m3["base_curve"]
    with open(RESULTS / "m1_sku_table.csv", encoding="utf-8") as f:
        a_rows = [r for r in csv.DictReader(f) if r["abc"] == "A"]
    a_low = sum(float(r["stockout_rate_d7"]) < 0.02 for r in a_rows)
    return {
        "target days": f"{t['days_of_cover']:.1f} 天",
        "target ci": f"{t['ci95'][0]:.1f}～{t['ci95'][1]:.1f}",
        "method days": f"{m['range_days']:.1f} 天",
        "method ci": f"{m['ci95'][0]:.1f}～{m['ci95'][1]:.1f}",
        "weekday wape excl": f"WAPE {ex['weekday_mean']['wape']:.3f}",
        "lgbm cal excl": f"假期 {ex['lgbm_calendar']['wape']:.3f}",
        "wins": f"{pe['n']} 週裡贏 {pe['a_wins']} 週，p = {pe['wilcoxon_p']:.2f}",
        "ma28": f"WAPE {m2['overall']['ma28']['wape']:.3f} vs {m2['overall']['weekday_mean']['wape']:.3f}，"
                f"p = {pa['wilcoxon_p']}",
        "ma28 losses": f"{pa['n']} 週全輸" if pa["a_wins"] == 0 else "MA28 有贏",
        "h2 yz": f"+{sf['YZ'] * 100:.1f} 個百分點",
        "h2 x": f"−{abs(sf['X']) * 100:.1f} 個百分點",
        "setups": f"從 {m4['actual_plan']['setups']} 降到 {m4['perfect_info_same_setup_cost']['setups']}",
        "backlog": f"{roll['no_safety_stock']['mean_backlog_t']:.0f} 噸／週",
        "cost roll": f"{roll['with_safety_stock']['cost']:,.0f}",
        "cost actual": f"{m4['actual_cost_same_setup_cost']:,.0f}",
        "pr logistic": f"{on['metrics']['logistic']['pr_auc']:.3f}",
        "pr rule": f"{on['metrics']['rule_age']['pr_auc']:.3f}",
        "boot ci": f"{boot['ci95'][0]:.3f}～{boot['ci95'][1]:.3f}",
        "alerts": f"{on['metrics']['logistic']['at_threshold']['alerts_per_day']:.0f} 個左右",
        "fill": f"{m1['fill_rate_ton_weighted']:.3f}",
        "dtf": f"中位 {fifo['median_days_to_fill']:.0f} 天、p90 {fifo['p90_days_to_fill']:.0f} 天",
        "otif": f"{pct(fifo['otif_proxy_by_d']['7']['by_ton'])}（以噸計）／{pct(fifo['otif_proxy_by_d']['7']['by_lot'])}（以批計）",
        "stockout": f"{fifo['stockout_rate_by_d']['7'] * 100:.1f}%",
        "a class": f"A 類 {len(a_rows)} 個 SKU 有 {a_low} 個低於 2%",
        "cover": f"中位 {m1['days_of_cover_lower_bound']['portfolio_median']:.1f} 天",
        "mc": f"{mc['p05']:.3f}～{mc['p95']:.3f}",
        "sim cover range": f"約 {base['0.8']['days_of_cover']:.0f}～{base['0.99']['days_of_cover']:.0f} 天",
        "random split": f"從 {m5['random_split_demo']['lightgbm_pr_auc_time_split']:.3f} 拉高到 "
                        f"{m5['random_split_demo']['lightgbm_pr_auc_random_split']:.3f}",
    }


@pytest.mark.parametrize("key,text", expected().items())
def test_readme_number_matches_results(key, text):
    assert text in README, f"{key}：README 找不到「{text}」"
