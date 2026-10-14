# ==========================================================
# 50 STARTUPS - CRISP-DM MULTIPLE LINEAR REGRESSION
#
# Model B: 5 Features + Intercept
#
# Feature Selection:
# 1. LassoCV
# 2. RFECV
# 3. Sequential Forward Selection (SFS)
# 4. ElasticNetCV
# 5. Exhaustive Feature Selection
#
# Plus: Baseline, VIF, CV, Metrics, Plots, Top 10 Outliers
#
# pip install numpy pandas matplotlib scikit-learn statsmodels
# ==========================================================

import itertools
import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 設定輸出編碼與繪圖設定
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_URL = "https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv"
BACKUP_URL = "https://raw.githubusercontent.com/datasets-repository/50-Startups/master/50_Startups.csv"
LOCAL_DATA_PATH = os.path.join(BASE_DIR, "50_Startups.csv")

# ==============================================================================
# Phase 1 & 2: Business & Data Understanding (商業與資料理解)
# ==============================================================================
print("=" * 70)
print("[Phase 1 & 2] Business & Data Understanding: 50 Startups Dataset")
print("=" * 70)

if os.path.exists(LOCAL_DATA_PATH):
    print(f"從本地快取載入資料: {LOCAL_DATA_PATH}")
    df = pd.read_csv(LOCAL_DATA_PATH)
else:
    print(f"正在從 GitHub 下載資料: {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        print(f"主網址下載失敗 ({e})，改用備用網址: {BACKUP_URL}")
        df = pd.read_csv(BACKUP_URL)
    df.to_csv(LOCAL_DATA_PATH, index=False)

print(f"成功載入資料！維度: {df.shape}")
print("\n--- 前 5 筆預覽 ---")
print(df.head())

print("\n--- 敘述統計摘要 ---")
print(df.describe())

# ==============================================================================
# Phase 3: Data Preparation (資料準備 - Model B: 5 Features + Intercept)
# ==============================================================================
print("\n" + "=" * 70)
print("[Phase 3] Data Preparation: Model B (5 Features + Intercept)")
print("=" * 70)

# 進行虛擬變數編碼 (One-Hot Encoding drop_first 避免虛擬變數陷阱)
X = pd.get_dummies(df.drop(columns=["Profit"]), columns=["State"], drop_first=True, dtype=float)
y = df["Profit"]

feature_names = list(X.columns)
print(f"Model B 特徵清單 (共 {len(feature_names)} 個特徵): {feature_names}")

# 資料切分: 80% 訓練集, 20% 測試集 (固定 random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"訓練集筆數: {len(X_train)} 筆, 測試集筆數: {len(X_test)} 筆")

# 5-Fold 交叉驗證生成器 (Shuffle=True, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 標準化縮放 (以訓練集為標準)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

# 變異數膨脹因子 (VIF) 檢測多重共線性 (Multicollinearity)
print("\n--- 多重共線性診斷 (Variance Inflation Factor, VIF) ---")
X_train_vif = sm.add_constant(X_train)
vif_df = pd.DataFrame({
    "Feature": X_train_vif.columns,
    "VIF": [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]
})
print(vif_df.to_string(index=False))

# ==============================================================================
# Phase 4: Modeling & Feature Selection (5 大特徵挑選演算法)
# ==============================================================================
print("\n" + "=" * 70)
print("[Phase 4] Model Training & Feature Selection (Baseline vs 5 Methods)")
print("=" * 70)

results = []

# --- 0. Baseline OLS (全 5 特徵) ---
baseline_ols = LinearRegression(fit_intercept=True)
baseline_ols.fit(X_train, y_train)
baseline_cv_rmse = -cross_val_score(baseline_ols, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()
baseline_test_r2 = r2_score(y_test, baseline_ols.predict(X_test))

results.append({
    "Method": "Baseline OLS",
    "Selected_Features": feature_names,
    "Num_Features": len(feature_names),
    "CV_RMSE": baseline_cv_rmse,
    "Test_R2": baseline_test_r2
})

# --- 1. LassoCV (L1 正則化) ---
lasso = LassoCV(cv=cv, random_state=42).fit(X_train_scaled, y_train)
lasso_selected = [col for col, coef in zip(feature_names, lasso.coef_) if abs(coef) > 1e-4]
lr_lasso = LinearRegression().fit(X_train[lasso_selected], y_train)
lasso_cv_rmse = -cross_val_score(lr_lasso, X_train[lasso_selected], y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()
lasso_test_r2 = r2_score(y_test, lr_lasso.predict(X_test[lasso_selected]))

results.append({
    "Method": "LassoCV",
    "Selected_Features": lasso_selected,
    "Num_Features": len(lasso_selected),
    "CV_RMSE": lasso_cv_rmse,
    "Test_R2": lasso_test_r2
})

# --- 2. RFECV (遞迴特徵消除 + 交叉驗證) ---
rfecv = RFECV(estimator=LinearRegression(), cv=cv, scoring="neg_root_mean_squared_error", min_features_to_select=1)
rfecv.fit(X_train_scaled, y_train)
rfecv_selected = [col for col, sup in zip(feature_names, rfecv.support_) if sup]
lr_rfecv = LinearRegression().fit(X_train[rfecv_selected], y_train)
rfecv_cv_rmse = -cross_val_score(lr_rfecv, X_train[rfecv_selected], y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()
rfecv_test_r2 = r2_score(y_test, lr_rfecv.predict(X_test[rfecv_selected]))

results.append({
    "Method": "RFECV",
    "Selected_Features": rfecv_selected,
    "Num_Features": len(rfecv_selected),
    "CV_RMSE": rfecv_cv_rmse,
    "Test_R2": rfecv_test_r2
})

# --- 3. Sequential Forward Selection (Forward SFS) ---
sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=3, direction="forward", cv=cv, scoring="neg_root_mean_squared_error")
sfs.fit(X_train_scaled, y_train)
sfs_selected = [col for col, sup in zip(feature_names, sfs.get_support()) if sup]
lr_sfs = LinearRegression().fit(X_train[sfs_selected], y_train)
sfs_cv_rmse = -cross_val_score(lr_sfs, X_train[sfs_selected], y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()
sfs_test_r2 = r2_score(y_test, lr_sfs.predict(X_test[sfs_selected]))

results.append({
    "Method": "Forward SFS",
    "Selected_Features": sfs_selected,
    "Num_Features": len(sfs_selected),
    "CV_RMSE": sfs_cv_rmse,
    "Test_R2": sfs_test_r2
})

# --- 4. ElasticNetCV (彈性網絡正則化) ---
enet = ElasticNetCV(cv=cv, random_state=42, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]).fit(X_train_scaled, y_train)
enet_selected = [col for col, coef in zip(feature_names, enet.coef_) if abs(coef) > 1e-4]
lr_enet = LinearRegression().fit(X_train[enet_selected], y_train)
enet_cv_rmse = -cross_val_score(lr_enet, X_train[enet_selected], y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()
enet_test_r2 = r2_score(y_test, lr_enet.predict(X_test[enet_selected]))

results.append({
    "Method": "ElasticNetCV",
    "Selected_Features": enet_selected,
    "Num_Features": len(enet_selected),
    "CV_RMSE": enet_cv_rmse,
    "Test_R2": enet_test_r2
})

# --- 5. Exhaustive Feature Selection (窮舉所有特徵組合) ---
best_ex_score = float("inf")
best_ex_combo = None
for k in range(1, len(feature_names) + 1):
    for combo in itertools.combinations(feature_names, k):
        score = -cross_val_score(LinearRegression(), X_train[list(combo)], y_train, cv=cv, scoring="neg_root_mean_squared_error").mean()
        if score < best_ex_score:
            best_ex_score = score
            best_ex_combo = list(combo)

lr_ex = LinearRegression().fit(X_train[best_ex_combo], y_train)
ex_test_r2 = r2_score(y_test, lr_ex.predict(X_test[best_ex_combo]))

results.append({
    "Method": "Exhaustive",
    "Selected_Features": best_ex_combo,
    "Num_Features": len(best_ex_combo),
    "CV_RMSE": best_ex_score,
    "Test_R2": ex_test_r2
})

# 整合並輸出特徵選擇比較表
df_results = pd.DataFrame(results)
csv_path = os.path.join(BASE_DIR, "feature_selection_comparison.csv")
df_results.to_csv(csv_path, index=False)
print("\n--- 特徵篩選比較結果 ---")
print(df_results[["Method", "Num_Features", "CV_RMSE", "Test_R2"]].to_string(index=False))
print(f"\n已匯出比較結果至: {csv_path}")

# ==============================================================================
# Phase 5: Evaluation & Diagnostic Plots (圖表繪製與離群值檢驗)
# ==============================================================================
print("\n" + "=" * 70)
print("[Phase 5] Evaluation, Diagnostics & Top 10 Outliers")
print("=" * 70)

# 1. 繪製特徵篩選比較長條圖 (橫向長條圖，精確對照老師版面)
plot_methods = ["LassoCV", "RFECV", "Forward SFS", "ElasticNetCV", "Exhaustive", "Baseline OLS"]
rmse_map = {row["Method"]: row["CV_RMSE"] for _, row in df_results.iterrows()}
plot_rmse = [rmse_map[m] for m in plot_methods]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(plot_methods, plot_rmse, color="#1f77b4", height=0.65)
ax.invert_yaxis()  # LassoCV 置頂，Baseline OLS 置底
ax.set_title("50 Startups: Feature Selection Comparison", fontsize=12, pad=10)
ax.set_xlabel("Cross-Validated RMSE (Lower is Better)", fontsize=10)
ax.set_xlim(0, 10000)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(True)

comparison_png = os.path.join(BASE_DIR, "feature_selection_comparison.png")
plt.tight_layout()
plt.savefig(comparison_png, dpi=300)
print(f"已儲存特徵選擇長條圖至: {comparison_png}")

# 2. Top 10 Outliers (庫克距離 Cook's Distance 檢驗影響力異常值)
ols_full = sm.OLS(y, sm.add_constant(X)).fit()
influence = ols_full.get_influence()
cooks_d = influence.cooks_distance[0]
top10_idx = np.argsort(cooks_d)[-10:][::-1]

print("\n--- Top 10 Outliers (依據 Cook's Distance 影響力排名) ---")
outliers_df = pd.DataFrame({
    "Sample_Index": top10_idx,
    "Cooks_D": cooks_d[top10_idx],
    "Profit": y.iloc[top10_idx].values,
    "Predicted": ols_full.predict(sm.add_constant(X)).iloc[top10_idx].values
})
print(outliers_df.to_string(index=False))

# 3. 繪製預測圖 (含 95% 信賴區間 / 預測區間與 Top 10 離群值標記)
y_all_pred = ols_full.predict(sm.add_constant(X))
residuals = y - y_all_pred
residual_std = np.std(residuals)

fig, ax = plt.subplots(figsize=(9, 6))
# 畫所有樣本
ax.scatter(y, y_all_pred, color="#1f77b4", alpha=0.7, label="Startups (Normal)")
# 標記前 10 大離群值
ax.scatter(y.iloc[top10_idx], y_all_pred.iloc[top10_idx], color="red", s=70, edgecolors="black", zorder=5, label="Top 10 Outliers (Cook's D)")

for idx in top10_idx[:5]:  # 標註前 5 大離群點 index
    ax.annotate(f"#{idx}", (y.iloc[idx], y_all_pred.iloc[idx]), textcoords="offset points", xytext=(5, 5), fontsize=8, color="darkred")

# 45 度理想線
min_val = min(y.min(), y_all_pred.min()) - 10000
max_val = max(y.max(), y_all_pred.max()) + 10000
line_pts = np.linspace(min_val, max_val, 100)
ax.plot(line_pts, line_pts, color="black", linestyle="--", linewidth=1.2, label="Ideal Fit (y = ŷ)")

# 95% 預測區間 (Prediction Interval: ±1.96 * Residual Std)
ax.plot(line_pts, line_pts + 1.96 * residual_std, color="gray", linestyle=":", label="95% Prediction Interval")
ax.plot(line_pts, line_pts - 1.96 * residual_std, color="gray", linestyle=":")
ax.fill_between(line_pts, line_pts - 1.96 * residual_std, line_pts + 1.96 * residual_std, color="gray", alpha=0.1)

ax.set_title("50 Startups: Actual vs. Predicted (with 95% Prediction Interval & Outliers)", fontsize=12)
ax.set_xlabel("Actual Profit ($)", fontsize=10)
ax.set_ylabel("Predicted Profit ($)", fontsize=10)
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)

outliers_png = os.path.join(BASE_DIR, "actual_vs_predicted_outliers.png")
plt.tight_layout()
plt.savefig(outliers_png, dpi=300)
print(f"已儲存預測區間與離群值圖至: {outliers_png}")

# ==============================================================================
# Phase 6: Deployment (最終模型管線儲存與新創團隊即時推論)
# ==============================================================================
print("\n" + "=" * 70)
print("[Phase 6] Deployment: Pipeline Export & Real-Time Inference")
print("=" * 70)

selected_features = ["R&D Spend", "Administration", "Marketing Spend"]
print(f"最終部署採納特徵 (5 大演算法一致選取): {selected_features}")

# 建立乾淨的端到端推論管線 (僅需輸入原始 3 大費用)
deploy_preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), selected_features)],
    remainder="drop"
)
final_pipeline = Pipeline(
    steps=[
        ("preprocessor", deploy_preprocessor),
        ("regressor", LinearRegression(fit_intercept=True))
    ]
)

final_pipeline.fit(df[selected_features], df["Profit"])

# 導出模型檔案
MODEL_PATH = os.path.join(BASE_DIR, "startup_profit_pipeline.joblib")
joblib.dump(final_pipeline, MODEL_PATH)
print(f"推論 Pipeline 已成功導出至: {MODEL_PATH}")

# 讀取並執行新資料推論
deployed_model = joblib.load(MODEL_PATH)
new_startups = pd.DataFrame([
    {"R&D Spend": 160000.0, "Administration": 130000.0, "Marketing Spend": 300000.0, "State": "New York"},
    {"R&D Spend": 75000.0, "Administration": 110000.0, "Marketing Spend": 140000.0, "State": "California"},
    {"R&D Spend": 100000.0, "Administration": 85000.0, "Marketing Spend": 205000.0, "State": "Florida"}
])

predictions = deployed_model.predict(new_startups)

print("\n--- 新樣本獲利預測結果 ---")
for i, (pred, (_, row)) in enumerate(zip(predictions, new_startups.iterrows()), start=1):
    print(
        f"團隊 {i} [{row['State']}] (研發: ${row['R&D Spend']:,.0f}, "
        f"行政: ${row['Administration']:,.0f}, 行銷: ${row['Marketing Spend']:,.0f}) "
        f"--> 預估獲利: ${pred:,.2f}"
    )

print("\n所有流程執行完畢！")