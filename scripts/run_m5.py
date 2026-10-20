"""M5：欠貨「新發生」風險分類，與規則基準比，附隨機切分的洩漏示範與警示清單。

    python scripts/run_m5.py      # 需先跑過 run_m2.py（讀有效 SKU 清單）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import classify as C  # noqa: E402
import data  # noqa: E402

RESULTS = ROOT / "results"
N_BOOT = 1000
SEED = 0


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def design_matrix(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    X = df[C.FEATURES].copy()
    for g in groups:
        X[f"group_{g}"] = (df.group == g).astype(float)
    return X


def fit_models(Xtr, ytr):
    import lightgbm as lgb
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(Xtr, ytr)
    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=SEED, deterministic=True,
                             force_row_wise=True, n_jobs=1, verbose=-1)
    gbm.fit(Xtr, ytr)
    return {"logistic": lr, "lightgbm": gbm}


def scores(models, X, df):
    out = {name: m.predict_proba(X)[:, 1] for name, m in models.items()}
    out["rule_age"] = df.age.to_numpy()
    out["rule_backlog_days"] = df.backlog_days.to_numpy()
    return out


def best_f1_threshold(y, p) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 200)))
    f1 = []
    for g in grid:
        pred = p >= g
        tp = (pred & (y == 1)).sum()
        prec = tp / max(pred.sum(), 1)
        rec = tp / max((y == 1).sum(), 1)
        f1.append(0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return float(grid[int(np.argmax(f1))])


def at_threshold(y, p, thr, days) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    return {"threshold": _r(thr), "precision": _r(tp / max(pred.sum(), 1)),
            "recall": _r(tp / max((y == 1).sum(), 1)),
            "alerts_per_day": _r(pred.sum() / days, 2)}


def main() -> None:
    flows, skus, _ = data.load()
    meta = skus.set_index("sku")
    active = sorted(pd.read_csv(RESULTS / "m2_backtest.csv").sku.unique())
    pv = lambda f: data.pivot(flows, f)[active]
    P = C.panel(pv("order"), pv("delivery"), pv("production"), pv("factory_issue"),
                meta.ton_per_unit, meta.group)
    groups = sorted(P.group.unique())
    train, test = C.time_split(P.t)

    out = {"design": {"label": "onset", "stockout_days": C.D_STOCKOUT, "horizon": C.HORIZON,
                      "train_t": [27, C.TRAIN_END], "purge": C.PURGE,
                      "test_t": [C.TRAIN_END + C.PURGE, C.TEST_END],
                      "test_dates": [str(P[test].date.min().date()), str(P[test].date.max().date())],
                      "features": C.FEATURES + [f"group_{g}" for g in groups]}}

    res = {}
    for label in ("onset", "any7"):
        tr = P[train & P[label].notna()]
        te = P[test & P[label].notna()]
        ytr, yte = tr[label].to_numpy(int), te[label].to_numpy(int)

        # 門檻在訓練期內選：擬合 [27,106)、驗證 [113,140)
        inner_fit = tr[tr.t < C.INNER_FIT_END]
        inner_val = tr[tr.t >= C.INNER_VAL_START]
        m_inner = fit_models(design_matrix(inner_fit, groups), inner_fit[label].to_numpy(int))
        s_val = scores(m_inner, design_matrix(inner_val, groups), inner_val)
        thr = {k: best_f1_threshold(inner_val[label].to_numpy(int), v) for k, v in s_val.items()}

        models = fit_models(design_matrix(tr, groups), ytr)
        s_te = scores(models, design_matrix(te, groups), te)
        n_days = te.t.nunique()
        metrics = {}
        for k, p in s_te.items():
            m = {"pr_auc": _r(average_precision_score(yte, p)), "roc_auc": _r(roc_auc_score(yte, p))}
            if k in models:
                m["brier"] = _r(brier_score_loss(yte, p))
            m["at_threshold"] = at_threshold(yte, p, thr[k], n_days)
            metrics[k] = m

        # 按日 bootstrap：LightGBM、邏輯迴歸各自減去最好的規則
        rng = np.random.default_rng(SEED)
        days = te.t.unique()
        best_rule = max(("rule_age", "rule_backlog_days"), key=lambda k: metrics[k]["pr_auc"])
        diffs = {k: [] for k in models}
        for _ in range(N_BOOT):
            pick = rng.choice(days, len(days), replace=True)
            idx = np.concatenate([np.flatnonzero(te.t.to_numpy() == d) for d in pick])
            yb = yte[idx]
            if yb.sum() == 0:
                continue
            base = average_precision_score(yb, s_te[best_rule][idx])
            for k in models:
                diffs[k].append(average_precision_score(yb, s_te[k][idx]) - base)
        boot = {k: {"vs": best_rule, "median_diff": _r(np.median(v)),
                    "ci95": [_r(np.percentile(v, 2.5)), _r(np.percentile(v, 97.5))],
                    "share_positive": _r(np.mean(np.array(v) > 0), 3)} for k, v in diffs.items()}

        res[label] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                      "prevalence_train": _r(ytr.mean()), "prevalence_test": _r(yte.mean()),
                      "metrics": metrics, "bootstrap_pr_auc_vs_best_rule": boot}
        if label == "onset":
            imp = pd.Series(models["lightgbm"].booster_.feature_importance("gain"),
                            index=design_matrix(tr, groups).columns)
            res[label]["lightgbm_gain_top8"] = {k: _r(v / imp.sum(), 3)
                                                for k, v in imp.sort_values(ascending=False).head(8).items()}
            coef = models["logistic"][-1].coef_[0]
            res[label]["logistic_coef_top8"] = {k: _r(v, 3) for k, v in pd.Series(
                coef, index=design_matrix(tr, groups).columns).sort_values(key=abs, ascending=False).head(8).items()}
            # 警示清單：測試期最後一天、今天沒缺貨的 SKU
            last = te[te.t == te.t.max()].copy()
            last["risk"] = s_te["lightgbm"][te.t.to_numpy() == te.t.max()]
            last["alert"] = last.risk >= thr["lightgbm"]
            last.sort_values("risk", ascending=False)[
                ["date", "sku", "group", "risk", "alert", "age", "backlog_days", "cover", "days_since_prod",
                 "trend", label]].round({"risk": 3, "age": 3, "backlog_days": 3, "cover": 3,
                                          "trend": 3}).to_csv(RESULTS / "m5_alerts_last_day.csv", index=False)

    # 洩漏示範：同樣特徵改用隨機切分
    pool = P[(train | test) & P.onset.notna()]
    rng = np.random.default_rng(SEED)
    mask = rng.random(len(pool)) < 0.7
    mr = fit_models(design_matrix(pool[mask], groups), pool[mask].onset.to_numpy(int))
    pr = mr["lightgbm"].predict_proba(design_matrix(pool[~mask], groups))[:, 1]
    res["random_split_demo"] = {"lightgbm_pr_auc_random_split": _r(average_precision_score(
        pool[~mask].onset.to_numpy(int), pr)), "lightgbm_pr_auc_time_split": res["onset"]["metrics"]["lightgbm"]["pr_auc"]}

    out.update(res)
    (RESULTS / "m5_classify.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for label in ("onset", "any7"):
        print(label, {k: (v["pr_auc"], v["roc_auc"]) for k, v in out[label]["metrics"].items()},
              "prev", out[label]["prevalence_test"])
        print("  boot", out[label]["bootstrap_pr_auc_vs_best_rule"])
        print("  thr", {k: v["at_threshold"] for k, v in out[label]["metrics"].items()})
    print(json.dumps({k: out["onset"][k] for k in ("lightgbm_gain_top8", "logistic_coef_top8")}, indent=1))
    print(out["random_split_demo"])


if __name__ == "__main__":
    main()
