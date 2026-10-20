"""M5 欠貨風險分類：特徵、標籤、時間切分。

特徵只能用第 t 天日終以前的資料。兩個容易漏掉的地方：
- 期初欠貨與期初庫存的「下限」是用整段期間算的，含未來資訊。特徵改用因果版本：
  FIFO 期初欠貨設 0；倉庫庫存的期初用「到第 t 天為止」的累計淨流量最低點。
- ABC×XYZ 分群也是整段期間算的，不放進特徵；改用近 28 天的滾動量與波動。
標籤是事後的真相，可以用整段期間的最佳估計（FIFO 期初欠貨下限版）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import forecast as F
import kpi

D_STOCKOUT = 7
HORIZON = 7
TRAIN_END, PURGE, TEST_END = 140, 7, 214
INNER_FIT_END, INNER_VAL_START = 106, 113
FEATURES = ["age", "backlog_days", "fill7", "fill28", "trend", "cv28", "vol28_t", "cover",
            "prod_ratio7", "issue_ratio7", "days_since_prod", "dow", "eid_next7"]


def stockout_matrix(order: pd.DataFrame, delivery: pd.DataFrame, opening: str) -> pd.DataFrame:
    """每個產品日是否缺貨（有訂單批等超過 D_STOCKOUT 天）。opening ∈ {"zero", "lower_bound"}。"""
    out = {}
    for s in order.columns:
        b0 = kpi.opening_lower_bound(order[s], delivery[s]) if opening == "lower_bound" else 0.0
        r = kpi.fifo(order[s].to_numpy(), delivery[s].to_numpy(), b0)
        out[s] = kpi.stockout_flags(r.oldest_open_age, D_STOCKOUT)
    return pd.DataFrame(out, index=order.index)


def labels(stock: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """now[t]＝今天缺貨；any7[t]＝第 t+1…t+7 天任一天缺貨（看不到完整 7 天的列為 NaN）。"""
    s = stock.astype(float)
    any7 = s.shift(-1).fillna(0)
    for k in range(2, HORIZON + 1):
        any7 = np.maximum(any7, s.shift(-k).fillna(0))
    any7.iloc[len(s) - HORIZON:] = np.nan
    return s, any7


def _roll(x: np.ndarray, w: int, fn: str) -> np.ndarray:
    r = pd.Series(x).rolling(w, min_periods=w)
    return getattr(r, fn)().to_numpy()


def features(order: pd.DataFrame, delivery: pd.DataFrame, production: pd.DataFrame,
             issue: pd.DataFrame, tpu: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """長表：每列一個 (t, sku)。全部特徵在第 t 天日終可算。"""
    dates = order.index
    cal = F.calendar(dates)
    eid = cal.eid_block.to_numpy()
    eid_next = np.array([eid[t + 1: t + 1 + HORIZON].max() if t + 1 < len(eid) else 0
                         for t in range(len(eid))])
    frames = []
    for s in order.columns:
        o, d = order[s].to_numpy(float), delivery[s].to_numpy(float)
        p, i = production[s].to_numpy(float), issue[s].to_numpy(float)
        r = kpi.fifo(o, d, 0.0)                                   # 因果版：期初欠貨 0
        m28 = _roll(o, 28, "mean")
        m7 = _roll(o, 7, "mean")
        sd28 = _roll(o, 28, "std")
        d7, d28 = _roll(d, 7, "sum"), _roll(d, 28, "sum")
        o7, o28 = _roll(o, 7, "sum"), _roll(o, 28, "sum")
        net = np.cumsum(i - d)
        stock = net - np.minimum(np.minimum.accumulate(net), 0)   # 期初＝到 t 為止的最低點
        last_prod = pd.Series(np.where(p > 0, np.arange(len(p)), np.nan)).ffill().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            f = pd.DataFrame({
                "t": np.arange(len(o)), "date": dates, "sku": s, "group": groups[s],
                "age": np.nan_to_num(r.oldest_open_age, nan=0.0),
                "backlog_days": np.where(m28 > 0, r.open_qty / m28, 0.0),
                "fill7": np.where(o7 > 0, np.minimum(d7 / o7, 1), 1.0),
                "fill28": np.where(o28 > 0, np.minimum(d28 / o28, 1), 1.0),
                "trend": np.where(m28 > 0, m7 / m28, 1.0),
                "cv28": np.where(m28 > 0, sd28 / m28, 0.0),
                "vol28_t": o28 * tpu[s],
                "cover": np.where(m28 > 0, stock / m28, 0.0),
                "prod_ratio7": np.where(o7 > 0, np.minimum(_roll(p, 7, "sum") / o7, 5), 0.0),
                "issue_ratio7": np.where(o7 > 0, np.minimum(_roll(i, 7, "sum") / o7, 5), 0.0),
                "days_since_prod": np.minimum(np.nan_to_num(np.arange(len(p)) - last_prod, nan=60), 60),
                "dow": dates.dayofweek, "eid_next7": eid_next,
            })
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def panel(order, delivery, production, issue, tpu, groups) -> pd.DataFrame:
    """特徵＋標籤。now／any7 用期初欠貨下限版（真相的最佳估計），onset＝今天沒缺、7 天內會缺。"""
    X = features(order, delivery, production, issue, tpu, groups)
    now, any7 = labels(stockout_matrix(order, delivery, "lower_bound"))
    X["now"] = now.stack().reindex(pd.MultiIndex.from_arrays([X.date, X.sku])).to_numpy()
    X["any7"] = any7.stack(future_stack=True).reindex(
        pd.MultiIndex.from_arrays([X.date, X.sku])).to_numpy()
    X["onset"] = np.where(X.now == 0, X.any7, np.nan)
    return X


def time_split(t: pd.Series, train_end: int = TRAIN_END, purge: int = PURGE,
               test_end: int = TEST_END, start: int = 27) -> tuple[pd.Series, pd.Series]:
    """訓練 [start, train_end)、清空 purge 天、測試 [train_end+purge, test_end)。

    訓練列 t 的標籤期是 t+1…t+HORIZON，必須早於第一個測試日。
    """
    if purge < HORIZON:
        raise ValueError("清空期短於標籤期，訓練標籤會與測試特徵重疊")
    train = (t >= start) & (t < train_end)
    test = (t >= train_end + purge) & (t < test_end)
    return train, test
