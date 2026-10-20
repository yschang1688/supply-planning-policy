# tasks — supply-planning-case

## M0 資料契約
- [x] T-001：`src/data.py` 下載＋sha256 驗證；不符即中止並指出檔名
- [x] T-002：寬表轉長表 `flows`＋SKU 主檔 `skus`＋`sku_sites`；件數／噸數對齊
- [x] T-003：`src/quality.py` 七項檢核（日期連續、負值、噸／件比固定、星期分布、出貨超前、期初下限、休眠 SKU）
- [x] T-004：每項檢核各有一則植入髒資料的反向測試

## M1 KPI 層
- [x] T-005：fill rate（上限 1、噸數加權彙總）
- [x] T-006：FIFO 帳齡、OTIF 代理曲線（D = 0…7）、期初欠貨 0 與下限兩版
- [x] T-007：缺貨率（排除休眠 SKU）
- [x] T-008：庫存天數（下限）
- [x] T-009：ABC × XYZ 分群
- [x] T-010：WAPE／bias（全 0 回 NaN）
- [x] T-011：每個 KPI 至少一則反向測試（例：欠貨符號寫反、FIFO 改 LIFO 時必紅）
- [x] T-012：`scripts/run_m0_m1.py` 落盤 `results/m0_quality.json`、`m1_kpi.json`、`m1_sku_table.csv`

## 基礎設施
- [x] T-013：`requirements.txt`、`.gitignore`、MIT LICENSE
- [x] T-014：GitHub Actions：tests（離線）＋claim-discipline

## M2 需求預測
- [x] T-020：M2 假期表（公開來源核對，附 URL）
- [x] T-021：M2 基準方法（季節 naive、星期平均、MA28）、ETS、Croston／TSB
- [x] T-022：M2 LightGBM 全域模型（特徵只用起點以前；有／無假期兩版）
- [x] T-023：M2 rolling-origin 回測與評估（噸數 WAPE／bias、分群、分 horizon、成對比較）
- [x] T-024：M2 反向測試（竄改未來值、固定週型態、Croston vs TSB 衰減）＋落盤結果宣稱守門

## M3 庫存政策模擬
- [x] T-030：模擬引擎（(R,S)、(s,S)、FIFO 欠單、隨機前置期）
- [x] T-031：安全庫存三種 q（normal_classic／normal／empirical），只用過去誤差
- [x] T-032：實驗格點（8 方法 × 6 目標 × 3 q × 2 政策 × L 掃描）＋Monte Carlo
- [x] T-033：H1、H2 判定與 bootstrap 信賴區間，落盤 m3_policy.json／m3_grid.csv
- [x] T-034：不變量測試（完美預測、守恆、SS 單調、前置期單調、不偷看未來）＋突變探針

## M4 產能排程 MILP
- [x] T-040：CLSP 模型（產能、開線連結、最小批量、期末條件、欠單開關、安全庫存軟下限），HiGHS
- [x] T-041：完全資訊取捨線（s 掃描）＋實際計畫定位
- [x] T-042：滾動計畫（星期平均預測、4 週期、s = 20；無／有安全庫存兩版）
- [x] T-043：測試（無解判定、限制遵守、放寬單調、s 單調、滾動不偷看）＋突變探針

## M5 欠貨風險分類
- [x] T-050：標籤（新發生／任何缺貨）與特徵（只用第 t 天以前）
- [x] T-051：時間切分＋清空期、訓練期內選門檻
- [x] T-052：規則基準、邏輯迴歸、LightGBM；PR-AUC 按日 bootstrap；隨機切分洩漏示範
- [x] T-053：警示清單落盤；測試（竄改未來、標籤、切分）＋突變探針

## M6 溝通
- [x] T-060：README 結論表、三句白話、現況 KPI 表、分析順序、限制段；總覽圖 `results/summary.png`
- [x] T-061：README 數字守門測試（`tests/test_readme_numbers.py`，CI claim-discipline 關）
- [x] T-062：humanizer 高信度全過
