# 50 Startups - CRISP-DM 多元線性迴歸與 5 大特徵選擇分析平台

本專案實作完整的資料科學 CRISP-DM 流程，針對著名的 **50 Startups 資料集** 進行深入的多元線性迴歸（Multiple Linear Regression）建模、統計診斷、5 大特徵篩選演算法比對，並建置高互動性的視覺化 Web 探索平台。

---

## 🌟 Demo 網站運行與造訪方式

本專案已包含完整的單頁互動式應用程式（SPA），支援即時特徵階層切換、Chart.js 動態折線圖、動態迴歸方程式、StatsUnlock 專業統計係數表、即時新創利潤預算機與 Top 10 離群值診斷。

### 啟動本地伺服器：
```bash
python -m http.server 8080
```
開啟瀏覽器造訪：**[http://localhost:8080/](http://localhost:8080/)**

---

## 📌 CRISP-DM 流程摘要

1. **Business Understanding (商業理解)**：
   - 探討研發 (R&D)、行銷 (Marketing)、行政管理 (Administration) 以及地理位置 (State) 對新創企業利潤 (Profit) 的邊際影響與資本配置效率。
2. **Data Understanding (資料理解)**：
   - 樣本共 50 筆，4 項自變數（3 連續數值 + 1 類別）與 1 項連續目標變數（Profit）。
3. **Data Preparation (資料準備 - Model B)**：
   - 採用 Model B 架構：保留截距項，將 3 個州別透過 One-Hot Encoding 轉換為 2 個虛擬變數（避免虛擬變數陷阱 Dummy Variable Trap），共 5 項特徵。
   - 進行 80/20 資料切分與 StandardScaler 標準化。
   - 計算變異數膨脹因子 (VIF)，確認無嚴重多重共線性。
4. **Modeling (模型建立與 5 大特徵篩選)**：
   - 實作 5 大特徵篩選演算法：`Lasso (L1)`、`Forward SFS`、`RFE`、`ElasticNet`、`Exhaustive Search`。
5. **Evaluation (模型評估與統計診斷)**：
   - 評估 $k=1 \dots 5$ 階層之 RMSE 與 $R^2$ 走勢。
   - 透過庫克距離 (Cook's Distance) 檢驗前 10 大離群值與極端樣本。
6. **Deployment (模型部署與線上推論)**：
   - 將最佳管線導出為 `startup_profit_pipeline.joblib`，並提供線上預測介面。

---

## 📊 5 大特徵選擇演算法成效對照表

| 特徵數量 ($k$) | 特徵子集組合 | 測試集 RMSE | 決定係數 $R^2$ | 模型評價 |
| :---: | :--- | :---: | :---: | :--- |
| **$k = 1$** | `R&D Spend` | **$8,274.87** | **0.9465** | 研發單一核心變數即解釋 94.65% 獲利變異 |
| **$k = 2$** | `R&D Spend` + `Marketing Spend` | **$8,198.80** | **0.9474** | **★ 全域最佳甜蜜點 (Global Sweet Spot)** |
| **$k = 3$** | + `State_New York` | **$8,309.06** | **0.9460** | 地理位置引入微幅雜訊，RMSE 上升 |
| **$k = 4$** | + `State_Florida` | **$8,409.92** | **0.9447** | 加入兩州虛擬變數，泛化能力進一步下滑 |
| **$k = 5$** | + `Administration` (全特徵) | **$9,137.99** | **0.9347** | **⚠ 過度擬合斷崖 (Overfitting Cliff)** |

---

## 💡 奧坎剃刀原則 (Occam's Razor / Parsimony)

> *“Entities should not be multiplied beyond necessity.” (若無必要，勿增實體)*

分析證實：**僅包含 2 個特徵（`R&D Spend`, `Marketing Spend`）的模型是全域最佳解**。
1. **RMSE 最低 ($8,198.80)**，優於 5 特徵模型的 $9,137.99（減少約 $939 泛化誤差）。
2. **$R^2$ 最高 (0.9474)**。
3. 行政管理費用（Administration）與利潤的相關性極低，納入模型僅會帶來雜訊與過擬合懲罰。
4. 在控制研發與行銷預算下，新創企業設於加州、佛羅里達或紐約對淨利無顯著獨立差異。

---

## 🙏 鳴謝與參考來源 (Acknowledgments)

- **設計與版面參考**：[StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/)
- **資料集來源**：50 Startups Dataset (Avik Jain 100-Days-Of-ML-Code)
- **開源函式庫**：`scikit-learn`, `statsmodels`, `pandas`, `numpy`, `matplotlib`, `chart.js`
