# openspec — 規格索引

流程：requirements（為什麼、驗收情境）→ design（決策與「為什麼不是另一個」）→ tasks（每條可獨立驗證）→ 實作 → 測試綠才勾。

## 進行中

| 變更 | 級 | 狀態 |
|---|---|---|
| [supply-planning-case](changes/supply-planning-case/) | L2（新 repo、跨多模組、新資料模型） | M0／M1 實作中；M2–M6 規格已列、細節開工前補 design 附錄 |

## 已歸檔

（無）

## 規則

1. 情境用 WHEN／THEN 寫，每條對應至少一個測試；只測 happy path 不算完成，每組要有「邏輯寫反就會紅」的反向測試。
2. 實作偏離 design 時，先回來改 design 再改程式，兩者落差不得超過一個模組。
3. 數字只寫已落盤在 `results/` 的值；README 的每個結論都要能在 `results/*.json` 找到出處，CI 的 claim-discipline 守這一條。
4. 完成的變更搬到 `changes/archive/YYYY-MM-DD-<name>/`，不刪。
