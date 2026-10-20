"""README 首屏的總覽圖：results/summary.png。只讀已落盤結果，不重算。

    python scripts/plot_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def main() -> None:
    m3 = json.loads((RESULTS / "m3_policy.json").read_text(encoding="utf-8"))
    m4 = json.loads((RESULTS / "m4_schedule.json").read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1, 1.25]})

    # ── 左：H1 ── 換服務目標 vs 換預測方法
    h1 = m3["h1"]
    t = h1["target_effect_0.80_to_0.99"]
    m = h1["method_effect_cover_at_equal_fill"]
    rows = [("Change forecasting method\n(at equal fill rate 0.98)", m["range_days"], m["ci95"]),
            ("Raise service target\n0.80 → 0.99", t["days_of_cover"], t["ci95"])]
    for i, (label, v, ci) in enumerate(rows):
        a.barh(i, v, height=0.3, color=BLUE)
        a.plot(ci, [i, i], color=INK2, lw=1.2)
        for x in ci:
            a.plot([x, x], [i - 0.08, i + 0.08], color=INK2, lw=1.2)
        a.text(ci[1] + 1, i, f"{v:.1f} days", va="center", color=INK, fontsize=10)
    a.set_yticks(range(len(rows)), [r[0] for r in rows])
    a.set_xlabel("Extra days of inventory (95% bootstrap CI)")
    a.set_xlim(0, 60)
    a.set_title("Service target vs. forecasting model",
                loc="left", fontsize=11, color=INK)

    # ── 右：M4 ── 換線 × 庫存取捨線
    fr = sorted(m4["frontier"], key=lambda r: r["setups"])
    b.plot([r["setups"] for r in fr], [r["mean_inventory_t"] for r in fr], color=BLUE, lw=2,
           marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    b.text(fr[0]["setups"] + 4, fr[0]["mean_inventory_t"] + 250,
           "MILP frontier\n(perfect demand information)", color=INK2, fontsize=9)
    act = m4["actual_plan"]
    roll = m4["rolling"]["with_safety_stock"]
    for pt, color, label, dx, ha in ((act, ORANGE, "Actual plan", -8, "right"),
                                     (roll, AQUA, "MILP on forecasts\n+ safety stock", 8, "left")):
        b.scatter(pt["setups"], pt["mean_inventory_t"], s=70, color=color, zorder=3,
                  edgecolor=SURFACE, linewidth=1.5)
        b.text(pt["setups"] + dx, pt["mean_inventory_t"], label, color=INK, fontsize=9,
               ha=ha, va="center")
    b.set_xlabel("Production setups (SKU-weeks with a run, 11 weeks)")
    b.set_ylabel("Mean inventory (tonnes)")
    b.set_ylim(0, 6000)
    b.set_xlim(90, 310)
    b.set_title("Changeovers vs. inventory (MILP, HiGHS)",
                loc="left", fontsize=11, color=INK)

    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x" if ax is a else "both", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(w_pad=3)
    fig.savefig(RESULTS / "summary.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
