"""資料契約：下載 SupplyGraph、驗 sha256、把四條日流量轉成一張長表。

來源：Wasi, Islam & Akib (2024), "SupplyGraph: A Benchmark Dataset for Supply
Chain Planning using Graph Neural Networks", arXiv:2401.15299。
上游 repo 授權為 LGPL-2.1，所以本 repo 不收原檔，只釘住 commit 與 sha256，
需要時重抓到 data/raw/（gitignored）。

這一層只做「讀進來、對齊、驗身分」，不修資料。資料的怪處交給 quality.py 報告。
"""
from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

UPSTREAM_COMMIT = "0e6d010e256ee08e34ab6bef5a111ef1c758b32d"
_BASE = f"https://raw.githubusercontent.com/CIOL-SUST/SupplyGraph/{UPSTREAM_COMMIT}/Raw Dataset/Homogenoeus/"

# 本地檔名 → (上游相對路徑, sha256)。上游檔名有尾端空白與大小寫不一致，照抄不要「修」。
FILES = {
    "unit_order.csv": ("Temporal Data/Unit/Sales Order.csv",
                       "c9fe75725c71b473be12c8610ca24efb26c6ca68173a1d7eb4af946429cdc95e"),
    "unit_delivery.csv": ("Temporal Data/Unit/Delivery To distributor.csv",
                          "c619763925212421c72630db59fb9e07808345cd1cc47f254fb96a0edebcfb85"),
    "unit_production.csv": ("Temporal Data/Unit/Production .csv",
                            "2098a1280d014c90416545531f97ccc8f836607de120b0a101310204c8f8079b"),
    "unit_factory_issue.csv": ("Temporal Data/Unit/Factory Issue.csv",
                               "2715e35a19a236899e48f485a1fc89569f1586fe53f6c5574935d342ecca6b63"),
    "ton_order.csv": ("Temporal Data/Weight/Sales Order .csv",
                      "fc466626f0cae410cfc884772521abad2ee6e39cd559cfee36f8677742aee7b7"),
    "ton_delivery.csv": ("Temporal Data/Weight/Delivery to Distributor.csv",
                         "75f90062f497b402d2d202a70d5c9368568134ab0b982ca38a650605a29ee3cb"),
    "ton_production.csv": ("Temporal Data/Weight/Production .csv",
                           "c54c3f190ca13888fba08e6e24c2bbb9711fe4faa52247518310c241eec1cc60"),
    "ton_factory_issue.csv": ("Temporal Data/Weight/Factory Issue.csv",
                              "1299e457d4d7590f337401e96f4c64f3e740ebe21df100a1d78c24b857b677d5"),
    "sku_groups.csv": ("Nodes/Node Types (Product Group and Subgroup).csv",
                       "949a75f6636f7c2917f655592d17abca7dfcd0a9597c838431a5df7b02fbfc31"),
    "sku_sites.csv": ("Nodes/Nodes Type (Plant & Storage).csv",
                      "da3ed29e0cfe80c3147e4e688f016bf0bd2deab315435666ddef2454e0a1685e"),
}

FLOWS = ("order", "production", "factory_issue", "delivery")


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(raw_dir: Path = RAW, files: dict = FILES) -> None:
    """每個檔案都要存在且 sha256 相符，否則中止並指出是哪一個。"""
    for name, (_, want) in files.items():
        p = raw_dir / name
        if not p.exists():
            raise FileNotFoundError(f"{name} 不存在；先跑 download()")
        got = sha256_of(p)
        if got != want:
            raise ChecksumError(f"{name} 的 sha256 不符：預期 {want[:12]}…，實得 {got[:12]}…")


def download(raw_dir: Path = RAW) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (rel, _) in FILES.items():
        dest = raw_dir / name
        if not dest.exists():
            url = _BASE + rel
            urllib.request.urlretrieve(urllib.parse.quote(url, safe=":/"), dest)
    verify(raw_dir)


def read_wide(path: Path) -> pd.DataFrame:
    """上游格式：第一欄 Date，其餘每欄一個 SKU，值為當日量。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").astype(float)


def wide_to_long(unit: dict[str, pd.DataFrame], ton: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """四條流量 × 兩種單位 → 長表 date, sku, flow, qty_unit, qty_ton。

    件數版與噸數版的日期與 SKU 必須完全一致；不一致代表兩份不是同一批資料，直接中止。
    """
    parts = []
    for flow in FLOWS:
        u, t = unit[flow], ton[flow]
        if not u.index.equals(t.index) or list(u.columns) != list(t.columns):
            raise ValueError(f"{flow}：件數版與噸數版的日期或 SKU 不對齊")
        lu = u.stack().rename("qty_unit")
        lt = t.stack().rename("qty_ton")
        part = pd.concat([lu, lt], axis=1).reset_index()
        part.columns = ["date", "sku", "qty_unit", "qty_ton"]
        part["flow"] = flow
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out[["date", "sku", "flow", "qty_unit", "qty_ton"]].sort_values(
        ["flow", "sku", "date"], ignore_index=True)


def load(raw_dir: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """回傳 (flows 長表, skus 主檔, sku_sites 對照)。載入前先驗 sha256。"""
    verify(raw_dir)
    unit = {f: read_wide(raw_dir / f"unit_{f}.csv") for f in FLOWS}
    ton = {f: read_wide(raw_dir / f"ton_{f}.csv") for f in FLOWS}
    flows = wide_to_long(unit, ton)

    groups = pd.read_csv(raw_dir / "sku_groups.csv").rename(
        columns={"Node": "sku", "Group": "group", "Sub-Group": "subgroup"})
    # 上游有兩個不同的 SKU 共用代碼 POP001L12P：流量檔把第二個寫成 "POP001L12P.1"
    # （四條流量的數值都不同，不是重複列），主檔則列了兩次同名。照流量檔的命名對齊。
    dup = groups["sku"].duplicated()
    groups.loc[dup, "sku"] = groups.loc[dup, "sku"] + ".1"
    # 噸／件換算係數：每個 SKU 一個固定值，取兩種單位都非 0 的列的中位數。
    # 上游噸數版有缺欄（例：POV005L04P 的工廠出庫噸數整欄為 0），所以 qty_ton 一律
    # 由件數 × 係數重算；上游原值保留在 qty_ton_reported，交給品質檢核比對。
    both = flows[(flows.qty_unit > 0) & (flows.qty_ton > 0)]
    tpu = (both.qty_ton / both.qty_unit).groupby(both.sku).median().rename("ton_per_unit")
    flows = flows.rename(columns={"qty_ton": "qty_ton_reported"})
    flows["qty_ton"] = flows.qty_unit * flows.sku.map(tpu)
    skus = groups.merge(tpu.reset_index(), on="sku", how="outer")

    sites = (pd.read_csv(raw_dir / "sku_sites.csv")
             .rename(columns={"Node": "sku", "Plant": "plant", "Storage Location": "storage_location"})
             .drop_duplicates(ignore_index=True))
    return flows, skus, sites


def pivot(flows: pd.DataFrame, flow: str, qty: str = "qty_unit") -> pd.DataFrame:
    """長表取回某一條流量的寬表（index=date、columns=sku），給 KPI 用。"""
    sub = flows[flows["flow"] == flow]
    return sub.pivot(index="date", columns="sku", values=qty).sort_index()
