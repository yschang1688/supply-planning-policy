# 供應鏈計畫：服務水準、庫存與產能，先定政策再談模型

[![tests](https://github.com/yschang1688/supply-planning-policy/actions/workflows/tests.yml/badge.svg)](https://github.com/yschang1688/supply-planning-policy/actions/workflows/tests.yml)

> **狀態：進行中。** M0 資料契約與 M1 KPI 層已完成；需求預測、庫存政策模擬、產能排程 MILP、欠貨風險分類開發中。結論段等全部模組完成後再寫，數字一律以 `results/*.json` 為準。

用一家製造商的真實日流量（訂單、生產、工廠出庫、出貨），回答計畫員每天要回答的三件事：服務水準目標怎麼定、要放多少庫存、產能怎麼排。

## 資料

SupplyGraph（Wasi, Islam & Akib, 2024, [arXiv:2401.15299](https://arxiv.org/abs/2401.15299)）：孟加拉一家 FMCG 製造商，41 SKU × 221 天（2023-01-01～2023-08-09），件數與噸數兩版。原始資料來自 [CIOL-SUST/SupplyGraph](https://github.com/CIOL-SUST/SupplyGraph)（LGPL-2.1），本 repo 不收原檔，`src/data.py` 釘住上游 commit 與每個檔案的 sha256，執行時重抓。

## 跑起來

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_m0_m1.py      # 下載、品質報告、KPI，落盤到 results/
.venv/bin/python -m pytest tests/ -q
```

## 目錄

| 你想看 | 去這裡 |
|---|---|
| 規格（為什麼、驗收情境、KPI 定義與取捨） | [`openspec/changes/supply-planning-case/`](openspec/changes/supply-planning-case/) |
| 資料下載、sha256、寬表轉長表 | [`src/data.py`](src/data.py) |
| 品質檢核（只報告、不修資料） | [`src/quality.py`](src/quality.py) → [`results/m0_quality.json`](results/m0_quality.json) |
| fill rate、FIFO 欠貨帳齡、OTIF 代理、缺貨率、庫存天數、ABC×XYZ | [`src/kpi.py`](src/kpi.py) → [`results/m1_kpi.json`](results/m1_kpi.json)、[`results/m1_sku_table.csv`](results/m1_sku_table.csv) |
