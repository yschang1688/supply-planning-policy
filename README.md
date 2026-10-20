# 供應鏈計畫：服務水準、庫存與產能，先定政策再談模型

[![tests](https://github.com/yschang1688/supply-planning-policy/actions/workflows/tests.yml/badge.svg)](https://github.com/yschang1688/supply-planning-policy/actions/workflows/tests.yml)

> **狀態：進行中。** M0 資料契約、M1 KPI 層、M2 需求預測回測、M3 庫存政策模擬、M4 產能排程 MILP 已完成；欠貨風險分類開發中。結論段等全部模組完成後再寫，數字一律以 `results/*.json` 為準。

用一家製造商的真實日流量（訂單、生產、工廠出庫、出貨），回答計畫員每天要回答的三件事：服務水準目標怎麼定、要放多少庫存、產能怎麼排。

## 資料

SupplyGraph（Wasi, Islam & Akib, 2024, [arXiv:2401.15299](https://arxiv.org/abs/2401.15299)）：孟加拉一家 FMCG 製造商，41 SKU × 221 天（2023-01-01～2023-08-09），件數與噸數兩版。原始資料來自 [CIOL-SUST/SupplyGraph](https://github.com/CIOL-SUST/SupplyGraph)（LGPL-2.1），本 repo 不收原檔，`src/data.py` 釘住上游 commit 與每個檔案的 sha256，執行時重抓。

## 跑起來

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_m0_m1.py      # 下載、品質報告、KPI，落盤到 results/
.venv/bin/python scripts/run_m2.py         # 15 個起點的預測回測（約 1 分鐘）
.venv/bin/python scripts/run_m3.py         # 庫存政策格點＋bootstrap＋Monte Carlo（約 9 分鐘）
.venv/bin/python scripts/run_m4.py         # 產能排程 MILP 取捨線＋滾動計畫（約 10 秒）
.venv/bin/python -m pytest tests/ -q
```

## 目錄

| 你想看 | 去這裡 |
|---|---|
| 規格（為什麼、驗收情境、KPI 定義與取捨） | [`openspec/changes/supply-planning-case/`](openspec/changes/supply-planning-case/) |
| 資料下載、sha256、寬表轉長表 | [`src/data.py`](src/data.py) |
| 品質檢核（只報告、不修資料） | [`src/quality.py`](src/quality.py) → [`results/m0_quality.json`](results/m0_quality.json) |
| fill rate、FIFO 欠貨帳齡、OTIF 代理、缺貨率、庫存天數、ABC×XYZ | [`src/kpi.py`](src/kpi.py) → [`results/m1_kpi.json`](results/m1_kpi.json)、[`results/m1_sku_table.csv`](results/m1_sku_table.csv) |
| 8 種預測方法、行事曆特徵、rolling-origin 回測 | [`src/forecast.py`](src/forecast.py) → [`results/m2_forecast.json`](results/m2_forecast.json)、[`results/m2_backtest.csv`](results/m2_backtest.csv) |
| (R,S)／(s,S) 逐日重播模擬、三種安全庫存、H1／H2 判定 | [`src/simulate.py`](src/simulate.py) → [`results/m3_policy.json`](results/m3_policy.json)、[`results/m3_grid.csv`](results/m3_grid.csv) |
| 週別有產能限制批量問題（CLSP）、換線 × 庫存取捨線、滾動計畫 | [`src/schedule.py`](src/schedule.py) → [`results/m4_schedule.json`](results/m4_schedule.json)、[`results/m4_frontier.csv`](results/m4_frontier.csv) |
