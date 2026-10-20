"""M2 需求預測：基準方法、間歇需求方法、ETS、LightGBM 全域模型，以及 rolling-origin 回測。

約定：`hist` 是到起點（含）為止的歷史，起點＝最後一個已知日；預測起點後 1…H 天。
任何函式都只能讀 `hist`，不得讀起點之後的資料——test_forecast.py 用竄改未來值直接驗證。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

H = 7

# ── 行事曆 ────────────────────────────────────────────────────────────
# 孟加拉 2023 年假期，來源 https://www.timeanddate.com/holidays/bangladesh/2023 （2026-10-20 擷取）。
# 只收資料期間（1–8 月）內的 Government Holiday；開齋節與宰牲節另把 Optional Holiday
# 的連假日併入 EID_BLOCKS，因為對經銷商下單而言整段都是休假。
GOV_HOLIDAYS = pd.to_datetime([
    "2023-02-21", "2023-03-08", "2023-03-17", "2023-03-26", "2023-04-14", "2023-04-19",
    "2023-04-20", "2023-04-21", "2023-04-22", "2023-04-23", "2023-05-01", "2023-05-04",
    "2023-06-28", "2023-06-29", "2023-06-30", "2023-07-19", "2023-07-29",
])
EID_BLOCKS = [("2023-04-20", "2023-04-24"), ("2023-06-27", "2023-06-30")]


def eid_days() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.concatenate(
        [pd.date_range(a, b).values for a, b in EID_BLOCKS]))


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """預測日的行事曆特徵；全部是事前公告的資訊。"""
    starts = pd.to_datetime([a for a, _ in EID_BLOCKS])
    since = np.full(len(dates), 15)
    for s in starts:
        d = (dates - s).days.to_numpy()
        ok = (d >= 0) & (d < 15)
        since[ok] = np.minimum(since[ok], d[ok])
    return pd.DataFrame({"dow": dates.dayofweek,
                         "gov_holiday": dates.isin(GOV_HOLIDAYS).astype(int),
                         "eid_block": dates.isin(eid_days()).astype(int),
                         "days_since_eid": since}, index=dates)


# ── 基準方法 ──────────────────────────────────────────────────────────
def seasonal_naive(hist: np.ndarray, h: int = H) -> np.ndarray:
    """上週同一天。h ≤ 7 時一定讀得到。"""
    hist = np.asarray(hist, float)
    return np.array([hist[len(hist) - 7 + (k % 7)] for k in range(h)])


def weekday_mean(hist: np.ndarray, h: int = H, weeks: int = 4) -> np.ndarray:
    """近 `weeks` 週同一星期的平均。"""
    hist = np.asarray(hist, float)
    n = len(hist)
    out = []
    for k in range(1, h + 1):
        idx = [n - 1 + k - 7 * w for w in range(1, weeks + 1)]
        idx = [i for i in idx if 0 <= i < n]
        out.append(hist[idx].mean())
    return np.array(out)


def moving_average(hist: np.ndarray, h: int = H, window: int = 28) -> np.ndarray:
    hist = np.asarray(hist, float)
    return np.full(h, hist[-window:].mean())


def ets(hist: np.ndarray, h: int = H) -> np.ndarray:
    """Holt-Winters 加法季節 7＋阻尼趨勢；失敗或全 0 時退回星期平均。"""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    hist = np.asarray(hist, float)
    if hist[-28:].sum() == 0:
        return np.zeros(h)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(hist, trend="add", damped_trend=True, seasonal="add",
                                       seasonal_periods=7, initialization_method="estimated").fit()
            fc = fit.forecast(h)
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite")
        return np.clip(fc, 0, None)
    except Exception:
        return weekday_mean(hist, h)


# ── 間歇需求 ──────────────────────────────────────────────────────────
def croston_rate(hist: np.ndarray, alpha: float = 0.1) -> float:
    """Croston：只在有需求的日子更新「需求量」與「間隔」，預測 = 量 ÷ 間隔。"""
    hist = np.asarray(hist, float)
    nz = np.flatnonzero(hist > 0)
    if len(nz) == 0:
        return 0.0
    z, p = hist[nz[0]], nz[0] + 1.0
    last = nz[0]
    for i in nz[1:]:
        z += alpha * (hist[i] - z)
        p += alpha * ((i - last) - p)
        last = i
    return float(z / p)


def tsb_rate(hist: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> float:
    """TSB：每一天都更新「有需求的機率」，所以需求停掉後預測會逐期衰減。"""
    hist = np.asarray(hist, float)
    nz = np.flatnonzero(hist > 0)
    if len(nz) == 0:
        return 0.0
    z, prob = hist[nz[0]], 1.0 / (nz[0] + 1)
    for y in hist[nz[0]:]:
        prob += beta * ((y > 0) - prob)
        if y > 0:
            z += alpha * (y - z)
    return float(prob * z)


def weekday_profile(hist: np.ndarray, start_dow: int, window: int = 28) -> np.ndarray:
    """起點前 window 天各星期的平均 ÷ 整體平均。start_dow＝hist 第一天的星期。"""
    hist = np.asarray(hist, float)
    n = len(hist)
    tail = hist[-window:]
    dows = (start_dow + np.arange(n - len(tail), n)) % 7
    overall = tail.mean()
    if overall <= 0:
        return np.ones(7)
    return np.array([tail[dows == d].mean() / overall if (dows == d).any() else 1.0
                     for d in range(7)])


def intermittent(hist: np.ndarray, start_dow: int, h: int = H, kind: str = "croston") -> np.ndarray:
    rate = croston_rate(hist) if kind == "croston" else tsb_rate(hist)
    prof = weekday_profile(hist, start_dow)
    n = len(hist)
    return rate * prof[(start_dow + n + np.arange(h)) % 7]


# ── LightGBM 全域模型 ─────────────────────────────────────────────────
FEATURES = ["dow", "h", "same_dow_1w", "same_dow_4w", "mean_7", "mean_28", "zero_share_28",
            "sku", "group"]
CAL_FEATURES = ["gov_holiday", "eid_block", "days_since_eid"]


def _rows(Y: np.ndarray, o: int, dates: pd.DatetimeIndex, skus, groups, cal: pd.DataFrame,
          h_max: int = H, last_known: int | None = None) -> pd.DataFrame:
    """以第 o 天為起點的特徵列（每個 SKU × h）。只讀 Y[: o+1]。

    last_known 給定時（訓練用），只保留目標日 ≤ last_known 的列並附上目標值。
    """
    past = Y[: o + 1]
    m28 = past[-28:].mean(axis=0)
    scale = np.where(m28 > 0, m28, np.maximum(past.mean(axis=0), 1e-6))
    m7 = past[-7:].mean(axis=0)
    z28 = (past[-28:] == 0).mean(axis=0)
    frames = []
    for h in range(1, h_max + 1):
        t = o + h
        if last_known is not None and t > last_known:
            break
        if t >= len(dates):
            break
        w4 = [t - 7 * w for w in range(1, 5) if t - 7 * w >= 0]
        f = pd.DataFrame({
            "sku": skus, "group": groups, "h": h, "dow": dates[t].dayofweek,
            "same_dow_1w": past[t - 7] / scale,
            "same_dow_4w": past[w4].mean(axis=0) / scale,
            "mean_7": m7 / scale, "mean_28": m28 / scale, "zero_share_28": z28,
            "scale": scale, "origin": o, "t": t,
        })
        for c in CAL_FEATURES:
            f[c] = cal[c].iloc[t]
        if last_known is not None:
            f["y"] = Y[t] / scale
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cat(df: pd.DataFrame, skus, groups) -> pd.DataFrame:
    df = df.copy()
    df["sku"] = pd.Categorical(df["sku"], categories=list(skus))
    df["group"] = pd.Categorical(df["group"], categories=sorted(set(groups)))
    return df


def lgbm_forecast(Y: np.ndarray, origin: int, dates: pd.DatetimeIndex, skus, groups,
                  use_calendar: bool, min_history: int = 28, seed: int = 0) -> np.ndarray:
    """回傳 shape (H, n_sku) 的預測。訓練列的目標日一律 ≤ origin。"""
    import lightgbm as lgb
    cal = calendar(dates)
    train = pd.concat([_rows(Y, o, dates, skus, groups, cal, last_known=origin)
                       for o in range(min_history - 1, origin)], ignore_index=True)
    test = _rows(Y, origin, dates, skus, groups, cal)
    feats = FEATURES + (CAL_FEATURES if use_calendar else [])
    model = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.05,
                              num_leaves=15, min_child_samples=20, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                              deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)
    model.fit(_cat(train, skus, groups)[feats], train["y"])
    pred = model.predict(_cat(test, skus, groups)[feats]) * test["scale"].to_numpy()
    return np.clip(pred, 0, None).reshape(-1, len(skus))


# ── 回測 ──────────────────────────────────────────────────────────────
METHODS = ("seasonal_naive", "weekday_mean", "ma28", "ets", "croston", "tsb",
           "lgbm", "lgbm_calendar")


def origins(n_days: int, first: int = 111, step: int = 7, h: int = H) -> list[int]:
    return list(range(first, n_days - h, step))


def backtest(order: pd.DataFrame, groups: pd.Series, origin_idx: list[int],
             methods=METHODS) -> pd.DataFrame:
    """回傳長表：origin, date, sku, h, actual, <每個方法一欄>。"""
    Y = order.to_numpy(float)
    dates = order.index
    skus = list(order.columns)
    grp = groups.reindex(skus).fillna("NA").to_numpy()
    start_dow = dates[0].dayofweek
    out = []
    for o in origin_idx:
        hist = Y[: o + 1]
        preds = {}
        for name in methods:
            if name == "lgbm":
                preds[name] = lgbm_forecast(Y, o, dates, skus, grp, use_calendar=False)
            elif name == "lgbm_calendar":
                preds[name] = lgbm_forecast(Y, o, dates, skus, grp, use_calendar=True)
            else:
                cols = []
                for j in range(len(skus)):
                    s = hist[:, j]
                    cols.append({
                        "seasonal_naive": lambda: seasonal_naive(s),
                        "weekday_mean": lambda: weekday_mean(s),
                        "ma28": lambda: moving_average(s),
                        "ets": lambda: ets(s),
                        "croston": lambda: intermittent(s, start_dow, kind="croston"),
                        "tsb": lambda: intermittent(s, start_dow, kind="tsb"),
                    }[name]())
                preds[name] = np.stack(cols, axis=1)
        for k in range(H):
            t = o + 1 + k
            row = pd.DataFrame({"origin": dates[o], "date": dates[t], "sku": skus, "h": k + 1,
                                "actual": Y[t]})
            for name in methods:
                row[name] = preds[name][k]
            out.append(row)
    return pd.concat(out, ignore_index=True)
