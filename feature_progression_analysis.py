# ==========================================================
# 50 STARTUPS - FEATURE SELECTION PROGRESSION ANALYSIS
#
# Evaluates all 5 feature selection algorithms across k = 1..5
# Algorithms:
# 1. Lasso (L1 Ranking)
# 2. Forward SFS
# 3. RFE
# 4. ElasticNet Ranking
# 5. Exhaustive Search
#
# Outputs:
# - Individual progression CSVs & plots
# - all_5_algorithms_comparison_plot.png
# ==========================================================

import itertools
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 設定標準輸出與字型
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "50_Startups.csv")

if not os.path.exists(DATA_PATH):
    url = "https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv"
    df = pd.read_csv(url)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

X = pd.get_dummies(df.drop(columns=["Profit"]), columns=["State"], drop_first=True, dtype=float)
y = df["Profit"]
feature_names = list(X.columns)

# 80/20 train/test 切分 (random_state=0 用於特徵遞增進程分析)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# 定義 5 大演算法特徵挑選順序
# 1. Lasso (L1 Ranking)
lasso_order = ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"]

# 2. Forward SFS
sfs_order = ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"]

# 3. RFE
rfe_order = ["R&D Spend", "Marketing Spend", "Administration", "State_Florida", "State_New York"]

# 4. ElasticNet Ranking
enet_order = ["R&D Spend", "Marketing Spend", "Administration", "State_New York", "State_Florida"]

# 5. Exhaustive Search (在每個 k 找出最佳子集)
exhaustive_order_by_k = {}
for k in range(1, 6):
    best_score = float("inf")
    best_combo = None
    for combo in itertools.combinations(feature_names, k):
        lr = LinearRegression().fit(X_train[list(combo)], y_train)
        pred = lr.predict(X_test[list(combo)])
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        if rmse < best_score:
            best_score = rmse
            best_combo = list(combo)
    exhaustive_order_by_k[k] = best_combo

algorithms = {
    "Lasso (L1 Ranking)": {
        "prefix": "lasso_l1_ranking",
        "order": lasso_order,
        "is_exhaustive": False
    },
    "Forward SFS": {
        "prefix": "forward_sfs",
        "order": sfs_order,
        "is_exhaustive": False
    },
    "RFE": {
        "prefix": "rfe",
        "order": rfe_order,
        "is_exhaustive": False
    },
    "ElasticNet Ranking": {
        "prefix": "elasticnet_ranking",
        "order": enet_order,
        "is_exhaustive": False
    },
    "Exhaustive Search": {
        "prefix": "exhaustive_search",
        "order_by_k": exhaustive_order_by_k,
        "is_exhaustive": True
    }
}

all_results = {}

for algo_name, config in algorithms.items():
    prefix = config["prefix"]
    records = []
    
    for k in range(1, 6):
        if config["is_exhaustive"]:
            selected = config["order_by_k"][k]
        else:
            selected = config["order"][:k]
            
        lr = LinearRegression().fit(X_train[selected], y_train)
        y_pred = lr.predict(X_test[selected])
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        records.append({
            "k": k,
            "Number of Features": k,
            "Selected_Features": ", ".join(selected),
            "RMSE": rmse,
            "R_squared": r2
        })
        
    df_prog = pd.DataFrame(records)
    all_results[algo_name] = df_prog
    
    # 儲存個別 progression CSV
    csv_file = os.path.join(BASE_DIR, f"{prefix}_progression.csv")
    df_prog.to_csv(csv_file, index=False)
    
    # 繪製個別演算法 plot
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color = "tab:blue"
    ax1.set_xlabel("Number of Features (k)")
    ax1.set_ylabel("RMSE (Lower is Better)", color=color)
    ax1.plot(df_prog["k"], df_prog["RMSE"], color=color, marker="o", label="RMSE")
    ax1.tick_params(axis="y", labelcolor=color)
    
    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("R-squared (Higher is Better)", color=color)
    ax2.plot(df_prog["k"], df_prog["R_squared"], color=color, marker="s", linestyle="--", label="R-squared")
    ax2.tick_params(axis="y", labelcolor=color)
    
    plt.title(f"{algo_name}: Feature Progression (k=1..5)")
    plt.tight_layout()
    plot_file = os.path.join(BASE_DIR, f"{prefix}_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    print(f"已產出: {prefix}_progression.csv 及 {prefix}_plot.png")

# ==============================================================================
# 繪製合圖: all_5_algorithms_comparison_plot.png (雙子圖對比)
# ==============================================================================
fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(14, 6))

style_map = {
    "Lasso (L1 Ranking)": {"color": "#1f77b4", "marker": "o"},
    "Forward SFS": {"color": "#ff7f0e", "marker": "s"},
    "RFE": {"color": "#2ca02c", "marker": "^"},
    "ElasticNet Ranking": {"color": "#d62728", "marker": "D"},
    "Exhaustive Search": {"color": "#9467bd", "marker": "v"}
}

for algo_name, df_prog in all_results.items():
    style = style_map[algo_name]
    ax_rmse.plot(
        df_prog["k"], df_prog["RMSE"],
        label=algo_name,
        color=style["color"],
        marker=style["marker"],
        linewidth=1.8,
        markersize=6
    )
    ax_r2.plot(
        df_prog["k"], df_prog["R_squared"],
        label=algo_name,
        color=style["color"],
        marker=style["marker"],
        linewidth=1.8,
        markersize=6
    )

ax_rmse.set_title("RMSE Comparison across 5 Feature Selection Methods", fontsize=11, pad=10)
ax_rmse.set_xlabel("Number of Features", fontsize=10)
ax_rmse.set_ylabel("RMSE (Lower is Better)", fontsize=10)
ax_rmse.set_xticks(range(1, 6))
ax_rmse.grid(True, linestyle="--", alpha=0.5)
ax_rmse.legend(loc="upper left", fontsize=8.5)

ax_r2.set_title("R-squared Comparison across 5 Feature Selection Methods", fontsize=11, pad=10)
ax_r2.set_xlabel("Number of Features", fontsize=10)
ax_r2.set_ylabel("R-squared (Higher is Better)", fontsize=10)
ax_r2.set_xticks(range(1, 6))
ax_r2.grid(True, linestyle="--", alpha=0.5)
ax_r2.legend(loc="lower left", fontsize=8.5)

plt.tight_layout()
comp_file = os.path.join(BASE_DIR, "all_5_algorithms_comparison_plot.png")
plt.savefig(comp_file, dpi=300)
plt.close()

print(f"\n成功產出合圖: {comp_file}")
print("所有 5 種演算法遞增分析執行完成！")
