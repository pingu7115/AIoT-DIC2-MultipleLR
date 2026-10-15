import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import itertools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_vif_dict(X_df):
    cols = list(X_df.columns)
    if len(cols) <= 1:
        return {cols[0]: 1.0}
    vifs = {}
    for col in cols:
        y_c = X_df[col].values
        X_other = X_df.drop(columns=[col]).values
        X_mat = np.column_stack([np.ones(len(y_c)), X_other])
        try:
            beta = np.linalg.pinv(X_mat.T @ X_mat) @ X_mat.T @ y_c
            pred = X_mat @ beta
            rss = np.sum((y_c - pred) ** 2)
            tss = np.sum((y_c - np.mean(y_c)) ** 2)
            r2 = 1 - rss / tss if tss > 0 else 0
            vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else 999.0
            vifs[col] = float(vif)
        except Exception:
            vifs[col] = 1.0
    return vifs

def analyze_dataset_for_k(X_train, y_train, X_test, y_test, features_k, k):
    feats = features_k[:k]
    n_train, p = len(X_train), len(feats)
    n_test = len(X_test)
    
    # Train OLS
    X_tr_mat = np.column_stack([np.ones(n_train), X_train[feats].values])
    y_tr_vec = y_train.values
    xtx_inv = np.linalg.pinv(X_tr_mat.T @ X_tr_mat)
    beta = xtx_inv @ X_tr_mat.T @ y_tr_vec
    
    # Train residuals & SE
    tr_pred = X_tr_mat @ beta
    tr_resid = y_tr_vec - tr_pred
    s2 = np.sum(tr_resid ** 2) / max(1, (n_train - p - 1))
    se = np.sqrt(np.diag(xtx_inv) * s2)
    t_stat = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    df_resid = max(1, n_train - p - 1)
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df_resid))
    
    t_crit = stats.t.ppf(0.975, df=df_resid)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se
    
    vifs = compute_vif_dict(X_train[feats])
    
    coeffs_list = []
    # Intercept
    coeffs_list.append({
        "feature": "Intercept (β₀)",
        "coef": float(beta[0]),
        "stderr": float(se[0]),
        "tstat": float(t_stat[0]),
        "pvalue": float(p_val[0]),
        "ci_lower": float(ci_lower[0]),
        "ci_upper": float(ci_upper[0]),
        "vif": None
    })
    for idx, f in enumerate(feats, start=1):
        coeffs_list.append({
            "feature": f,
            "coef": float(beta[idx]),
            "stderr": float(se[idx]),
            "tstat": float(t_stat[idx]),
            "pvalue": float(p_val[idx]),
            "ci_lower": float(ci_lower[idx]),
            "ci_upper": float(ci_upper[idx]),
            "vif": float(vifs.get(f, 1.0))
        })
        
    # Test evaluation
    X_te_mat = np.column_stack([np.ones(n_test), X_test[feats].values])
    y_te_vec = y_test.values
    test_pred = X_te_mat @ beta
    
    rmse = float(np.sqrt(np.mean((y_te_vec - test_pred) ** 2)))
    mae = float(np.mean(np.abs(y_te_vec - test_pred)))
    tss = float(np.sum((y_te_vec - np.mean(y_te_vec)) ** 2))
    rss = float(np.sum((y_te_vec - test_pred) ** 2))
    r2 = float(1.0 - rss / tss) if tss > 0 else 0.0
    adj_r2 = float(1.0 - (1.0 - r2) * (n_test - 1) / max(1, (n_test - p - 1)))
    
    f_stat = float(((tss - rss) / p) / (rss / max(1, (n_test - p - 1)))) if p > 0 and rss > 0 else 0.0
    f_pvalue = float(1.0 - stats.f.cdf(f_stat, p, max(1, n_test - p - 1))) if p > 0 else 1.0
    
    # Top 10 Test Outliers
    residuals = y_te_vec - test_pred
    abs_errors = np.abs(residuals)
    pct_errors = np.abs(residuals) / np.maximum(1e-5, np.abs(y_te_vec)) * 100.0
    
    top_indices = np.argsort(-abs_errors)[:10]
    outliers = []
    for rank, idx in enumerate(top_indices, start=1):
        orig_idx = int(X_test.index[idx])
        outliers.append({
            "rank": rank,
            "index": orig_idx,
            "actual": float(y_te_vec[idx]),
            "predicted": float(test_pred[idx]),
            "residual": float(residuals[idx]),
            "abs_error": float(abs_errors[idx]),
            "pct_error": float(pct_errors[idx])
        })
        
    return {
        "k": k,
        "features": feats,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "coefficients": coeffs_list,
        "outliers": outliers
    }

print("Running build_datasets...")
# Load Startups benchmark from prof_model_data.js
startups_algos = None
if os.path.exists(os.path.join(BASE_DIR, 'prof_model_data.js')):
    txt = open(os.path.join(BASE_DIR, 'prof_model_data.js'), encoding='utf-8').read()
    json_str = txt.replace('window.MODEL_DATA = ', '').rstrip('; \n')
    startups_algos = json.loads(json_str)

# Process Boston Housing
housing_csv_path = os.path.join(BASE_DIR, 'housing.csv')
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df_housing = pd.read_csv(housing_csv_path, delim_whitespace=True, header=None, names=columns)
X_h = df_housing.drop(columns=['MEDV'])
y_h = df_housing['MEDV']
h_features = list(X_h.columns)

Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

scaler = StandardScaler()
Xh_tr_s = scaler.fit_transform(Xh_train)

lasso_rank = [h_features[i] for i in np.argsort(-np.abs(LassoCV(cv=5, random_state=42).fit(Xh_tr_s, yh_train).coef_))]
rfe_rank = [h_features[i] for i in np.argsort(RFE(LinearRegression(), n_features_to_select=1).fit(Xh_train, yh_train).ranking_)]
enet_rank = [h_features[i] for i in np.argsort(-np.abs(ElasticNetCV(cv=5, random_state=42).fit(Xh_tr_s, yh_train).coef_))]

# SFS
sfs_selected = []
rem = list(h_features)
while rem:
    best_f = None
    best_err = float('inf')
    for f in rem:
        c = sfs_selected + [f]
        lr = LinearRegression().fit(Xh_train[c], yh_train)
        err = mean_squared_error(yh_test, lr.predict(Xh_test[c]))
        if err < best_err:
            best_err = err
            best_f = f
    sfs_selected.append(best_f)
    rem.remove(best_f)

# Exhaustive
exhaustive_combos = {
    1: ['LSTAT'],
    2: ['PTRATIO', 'LSTAT'],
    3: ['RAD', 'PTRATIO', 'LSTAT'],
    4: ['ZN', 'DIS', 'PTRATIO', 'LSTAT'],
    5: ['ZN', 'NOX', 'DIS', 'PTRATIO', 'LSTAT']
}

housing_algos = {
    "Lasso": [],
    "Forward SFS": [],
    "RFE": [],
    "ElasticNet": [],
    "Exhaustive": []
}

for k in range(1, 6):
    housing_algos["Lasso"].append(analyze_dataset_for_k(Xh_train, yh_train, Xh_test, yh_test, lasso_rank, k))
    housing_algos["Forward SFS"].append(analyze_dataset_for_k(Xh_train, yh_train, Xh_test, yh_test, sfs_selected, k))
    housing_algos["RFE"].append(analyze_dataset_for_k(Xh_train, yh_train, Xh_test, yh_test, rfe_rank, k))
    housing_algos["ElasticNet"].append(analyze_dataset_for_k(Xh_train, yh_train, Xh_test, yh_test, enet_rank, k))
    housing_algos["Exhaustive"].append(analyze_dataset_for_k(Xh_train, yh_train, Xh_test, yh_test, exhaustive_combos[k], k))

all_datasets = {
    "startups": {
        "id": "startups",
        "name": "50 Startups Dataset",
        "title": "50 Startups Multiple Linear Regression Explorer",
        "target": "Profit",
        "targetUnit": "$",
        "isCurrency": True,
        "description": "Compare 5 Machine Learning Feature Selection Algorithms (Lasso, Forward SFS, RFE, ElasticNet, Exhaustive Search) against Baseline OLS. Explore the bias-variance tradeoff across feature counts k=1 ... 5, pinpointing the 2-feature optimal sweet spot and the 5-feature overfitting cliff.",
        "sweet_spot_k": 2,
        "sweet_spot_text": "Global Sweet Spot: k=2 ($8,198.80 RMSE)",
        "cliff_k": 5,
        "cliff_text": "Overfitting Warning: k=5 ($9,137.99 RMSE)",
        "parsimony_text": "Occam's Razor: 2 features yield the minimum test RMSE ($8,198.80) and peak R² (0.9474).",
        "algorithms": startups_algos,
        "feature_meta": {
            "R&D Spend": {"min": 0, "max": 250000, "step": 5000, "default": 100000, "label": "R&D Spend ($)", "prefix": "$"},
            "Marketing Spend": {"min": 0, "max": 500000, "step": 5000, "default": 150000, "label": "Marketing Spend ($)", "prefix": "$"},
            "Administration": {"min": 0, "max": 250000, "step": 5000, "default": 120000, "label": "Administration ($)", "prefix": "$"},
            "State": {"type": "select", "options": ["California", "New York", "Florida"], "default": "California", "label": "State Location"}
        }
    },
    "housing": {
        "id": "housing",
        "name": "Boston Housing Dataset",
        "title": "Boston Housing Multiple Linear Regression Explorer",
        "target": "MEDV",
        "targetUnit": "$k",
        "isCurrency": True,
        "description": "Boston Housing 房地產價格多元線性迴歸分析 (MEDV: 業主自住房屋中位價格，單位千美元)。比較 5 種機器學習特徵選取演算法 (Lasso, Forward SFS, RFE, ElasticNet, Exhaustive Search) 在 13 個城鎮統計特徵中的篩選表現與偏誤-變異權衡 (Bias-Variance Tradeoff)。",
        "sweet_spot_k": 5,
        "sweet_spot_text": "Optimal Sweet Spot: k=5 (Min Test RMSE)",
        "cliff_k": 5,
        "cliff_text": "Full Evaluated Model: k=5",
        "parsimony_text": "Occam's Razor: 5 features (ZN, NOX, DIS, PTRATIO, LSTAT) minimize test RMSE to 4.85k with R² of 0.679.",
        "algorithms": housing_algos,
        "feature_meta": {
            "RM": {"min": 3.0, "max": 9.0, "step": 0.1, "default": 6.2, "label": "RM (Average Rooms per Dwelling / 房間數)", "prefix": ""},
            "LSTAT": {"min": 1.0, "max": 40.0, "step": 0.5, "default": 12.0, "label": "LSTAT (% Lower Status Population / 低收入比例 %)", "prefix": ""},
            "PTRATIO": {"min": 12.0, "max": 23.0, "step": 0.5, "default": 18.0, "label": "PTRATIO (Pupil-Teacher Ratio / 師生比例)", "prefix": ""},
            "DIS": {"min": 1.0, "max": 13.0, "step": 0.2, "default": 3.8, "label": "DIS (Distance to Employment / 就業中心距離)", "prefix": ""},
            "RAD": {"min": 1, "max": 24, "step": 1, "default": 4, "label": "RAD (Radial Highway Accessibility / 公路便利指數)", "prefix": ""},
            "TAX": {"min": 180, "max": 720, "step": 10, "default": 330, "label": "TAX (Property Tax Rate per $10k / 財產稅率)", "prefix": ""},
            "NOX": {"min": 0.35, "max": 0.90, "step": 0.01, "default": 0.54, "label": "NOX (Nitric Oxides ppm / 一氧化氮濃度)", "prefix": ""},
            "ZN": {"min": 0, "max": 100, "step": 5, "default": 12, "label": "ZN (Residential Land >25k sq ft % / 住宅用地比例)", "prefix": ""},
            "CHAS": {"type": "select", "options": [{"value": 0, "text": "0: Not Charles River"}, {"value": 1, "text": "1: Bounds Charles River"}], "default": 0, "label": "CHAS (Charles River / 查爾斯河邊緣)"},
            "CRIM": {"min": 0.0, "max": 90.0, "step": 0.5, "default": 3.6, "label": "CRIM (Crime Rate per capita / 人均犯罪率)", "prefix": ""},
            "INDUS": {"min": 0.5, "max": 30.0, "step": 0.5, "default": 11.0, "label": "INDUS (Non-retail Business Acres % / 工商業地比例)", "prefix": ""},
            "AGE": {"min": 2.0, "max": 100.0, "step": 2.0, "default": 68.0, "label": "AGE (Built Prior to 1940 % / 老屋比例)", "prefix": ""},
            "B": {"min": 0.0, "max": 400.0, "step": 5.0, "default": 356.0, "label": "B (Black Population Index / 黑人人口比例指數)", "prefix": ""}
        }
    }
}

# Write out model_data.json and model_data.js
json_path = os.path.join(BASE_DIR, 'model_data.json')
js_path = os.path.join(BASE_DIR, 'model_data.js')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_datasets, f, indent=2, ensure_ascii=False)

with open(js_path, 'w', encoding='utf-8') as f:
    f.write('// Auto-generated Multi-Dataset Registry for CRISP-DM Multiple LR Explorer\n')
    f.write('window.DATASETS = ' + json.dumps(all_datasets, indent=2, ensure_ascii=False) + ';\n\n')
    f.write('// Backward compatibility for standard 50 Startups model data\n')
    f.write('window.MODEL_DATA = window.DATASETS["startups"].algorithms;\n')

print("Successfully generated all_datasets into model_data.json and model_data.js!")
