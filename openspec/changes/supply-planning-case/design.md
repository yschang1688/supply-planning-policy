# design — supply-planning-case

## D1：資料來源與授權

- 來源：GitHub `CIOL-SUST/SupplyGraph`（Wasi et al., 2024, arXiv:2401.15299），`Raw Dataset/Homogenoeus/` 下的 Temporal Data（Unit／Weight 各四檔）與 Nodes。
- **不 vendor 原檔**：上游 repo 授權為 LGPL-2.1，且可穩定重抓。`src/data.py` 內釘每個檔案的 URL 與 sha256，下載到 `data/raw/`（gitignored）。
- 為什麼不用 DataCo（Mendeley，CC BY 4.0）：月筆數平到約 5,200、2017-10 腰斬，看起來是合成資料；`Late_delivery_risk` 完全由實際與預定運送天數決定，拿來做分類就是洩漏。為什麼不用 M5：零售、規模太大，且 Kaggle 需要帳號。

## D2：資料模型

長表 `flows`：`date (datetime64), sku (str), flow ∈ {order, production, factory_issue, delivery}, qty_unit (float), qty_ton (float)`。

- 上游欄名對映：`Sales Order` → order、`Production` → production、`Factory Issue` → factory_issue、`Delivery To distributor` → delivery。
- SKU 主檔 `skus`：`sku, group, subgroup, ton_per_unit`。`ton_per_unit` 取兩種單位都非 0 的列的噸／件比中位數；實測每個 SKU 的比值固定（相對差 < 1e-13），品質檢核守住這件事。
- **`qty_ton` 一律由件數 × `ton_per_unit` 重算**，上游原值留在 `qty_ton_reported`。原因：上游噸數版有缺欄（POV005L04P 的工廠出庫噸數整欄為 0）；整欄缺只報告，部分缺才算硬錯誤。
- **共用代碼**：上游有兩個不同的 SKU 都叫 POP001L12P，流量檔把第二個寫成 `POP001L12P.1`（四條流量數值都不同，不是重複列），主檔則列兩次同名。照流量檔命名對齊；`POP001L12P.1` 沒有廠別對照。
- 廠與儲位對照 `sku_sites`：去重後 277 列，一個 SKU 對 1–13 個廠。**生產量只有 SKU 層級、沒有分廠**，所以 M4 的產能限制不能逐廠設定（見 D6）。

## D3：品質檢核的定位

品質檢核輸出一份 dict（落盤 `results/m0_quality.json`），**只報告、不修資料**。理由：這份資料的「怪」大多是資訊不足（期初庫存、期初欠貨、週末），不是錯誤；修掉會把限制藏起來。硬錯誤只有：sha256 不符、日期不連續、負值、件數與噸數不對齊、噸／件比不固定、同一條流量部分缺噸數。遇到就中止。

已知怪處（下載後實測，結論以 `results/m0_quality.json` 為準）：週五訂單近 0（推測為當地週末）；約 11% 的產品日累計出貨超前累計訂單；倉庫流量（累計出庫 − 累計出貨）會掉到負值；約四分之一的 SKU 有 ≥ 98% 的天數為 0。

## D4：KPI 定義

| KPI | 定義 | 為什麼這樣定 | 為什麼不是另一個 |
|---|---|---|---|
| fill rate | min(Σ出貨, Σ訂單) ÷ Σ訂單（逐 SKU，再以噸數加權彙總） | 最直接的 in-full | 不用逐日平均：大量的日子與少量的日子權重會一樣 |
| OTIF 代理 | FIFO 把出貨分配給訂單批，D 天內交齊的批數比例；報 D = 0…7 曲線 | 沒有逐單交期，只能用「下單後幾天交齊」代理準時 | 不挑一個 D 當唯一答案：D 是假設，曲線才誠實 |
| 期初欠貨 | 下限 = max(0, max_t(累計出貨 − 累計訂單)) | 出貨超前訂單，代表期初有舊單在消化 | 不假設 0：會讓新訂單的延遲看起來比實際短 |
| 缺貨率 | 存在超過 D 天未交齊訂單批的產品日 ÷ 有效產品日 | 計畫員看得懂的「有沒有斷貨」 | 不用「庫存 = 0 的天數」：期初庫存未知，庫存只算得出下限 |
| 庫存天數 | （期初庫存下限 + 累計工廠出庫 − 累計出貨）÷ 近 28 天平均日訂單 | 與 DIO 同義、單位是天，非專家看得懂 | 不用金額版 DIO：沒有成本資料 |
| ABC × XYZ | ABC 用噸數累計 80%／95%；XYZ 用週需求 CV 0.5／1.0；休眠 = 零值日 ≥ 98% | 用噸數，不同包裝規格才能比；用週不用日，避開週五為 0 造成的假波動 | 不用件數：一件 2 公升和一件 25 公斤不可比 |
| WAPE／bias | WAPE = Σ\|e\| ÷ Σ實際；bias = Σ(預測 − 實際) ÷ Σ實際 | 間歇需求下 MAPE 會除以 0 | 實際全 0 時回 NaN，不回 0（回 0 會把沒有需求的 SKU 算成完美預測） |

## D5：程式結構

- `src/data.py`：下載、驗 sha256、讀寬表、轉長表、SKU 主檔。
- `src/quality.py`：純函式檢核，輸入長表，輸出 dict。
- `src/kpi.py`：純函式 KPI，輸入 numpy／pandas 序列，不碰檔案。
- `scripts/run_m0_m1.py`：串起來，寫 `results/m0_quality.json`、`results/m1_kpi.json`、`results/m1_sku_table.csv`。
- 測試只吃人工構造的小序列（手算得出答案），所以 CI 不需要資料、離線可跑。另有 claim-discipline 一關，檢查 README 的數字仍與 `results/*.json` 一致。

## D6：M2–M5 的待定事項（開工前補附錄）

- M4 產能：生產沒有分廠，先以「產品群」當產線代理，產能 = 該群歷史週產量（噸）的 p95。開工前用資料確認產品群與廠的對應是否夠乾淨；不夠乾淨就改成單一總產能。
- M3 前置期：從生產與出庫的時間差估，估不出來就用參數掃描（1–7 天）。
- M5 標籤：以 M1 的缺貨定義往前看 7 天，特徵只能用當天以前的資料。
