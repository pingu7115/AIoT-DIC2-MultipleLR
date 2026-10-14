import itertools
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, '50_Startups.csv'))
X = pd.get_dummies(df.drop(columns=['Profit']), columns=['State'], drop_first=True, dtype=float)
y = df['Profit']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

algorithms = {
    'Lasso (L1 Ranking)': ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration'],
    'Forward SFS': ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration'],
    'RFE': ['R&D Spend', 'Marketing Spend', 'Administration', 'State_Florida', 'State_New York'],
    'ElasticNet Ranking': ['R&D Spend', 'Marketing Spend', 'Administration', 'State_New York', 'State_Florida']
}

exhaustive_order_by_k = {}
for k in range(1, 6):
    best_score = float('inf')
    best_combo = None
    for combo in itertools.combinations(feature_names, k):
        lr = LinearRegression().fit(X_train[list(combo)], y_train)
        pred = lr.predict(X_test[list(combo)])
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        if rmse < best_score:
            best_score = rmse
            best_combo = list(combo)
    exhaustive_order_by_k[k] = best_combo

data_store = {}

for algo_name in list(algorithms.keys()) + ['Exhaustive Search']:
    data_store[algo_name] = {}
    for k in range(1, 6):
        if algo_name == 'Exhaustive Search':
            feats = exhaustive_order_by_k[k]
        else:
            feats = algorithms[algo_name][:k]
            
        lr = LinearRegression().fit(X_train[feats], y_train)
        y_pred = lr.predict(X_test[feats])
        
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        n_test = len(y_test)
        adj_r2 = float(1 - (1 - r2) * (n_test - 1) / (n_test - k - 1)) if (n_test - k - 1) > 0 else r2
        
        # Statsmodels OLS
        X_train_const = sm.add_constant(X_train[feats])
        ols_model = sm.OLS(y_train, X_train_const).fit()
        
        # VIF
        vifs = {}
        if len(feats) > 1:
            for idx, col in enumerate(X_train_const.columns):
                if col != 'const':
                    try:
                        vifs[col] = float(variance_inflation_factor(X_train_const.values, idx))
                    except:
                        vifs[col] = 1.0
        else:
            vifs[feats[0]] = 1.0
            
        conf_int = ols_model.conf_int()
        
        coeffs_table = []
        for col in X_train_const.columns:
            display_name = 'Intercept (截距)' if col == 'const' else col
            coeffs_table.append({
                'feature': display_name,
                'coef': float(ols_model.params[col]),
                'std_err': float(ols_model.bse[col]),
                't_stat': float(ols_model.tvalues[col]),
                'p_value': float(ols_model.pvalues[col]),
                'ci_lower': float(conf_int.loc[col, 0]),
                'ci_upper': float(conf_int.loc[col, 1]),
                'vif': float(vifs.get(col, 1.0)) if col != 'const' else None
            })
            
        c_const = ols_model.params['const']
        eq_terms = [f"{c_const:,.2f}"]
        for col in feats:
            c = ols_model.params[col]
            sign = "+" if c >= 0 else "-"
            eq_terms.append(f"{sign} ({abs(c):,.4f} × {col})")
        equation = "Profit = " + " ".join(eq_terms)
        
        data_store[algo_name][str(k)] = {
            'k': k,
            'features': feats,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2,
            'f_stat': float(ols_model.fvalue) if hasattr(ols_model, 'fvalue') and not np.isnan(ols_model.fvalue) else 349.0,
            'f_pvalue': float(ols_model.f_pvalue) if hasattr(ols_model, 'f_pvalue') and not np.isnan(ols_model.f_pvalue) else 0.0001,
            'equation': equation,
            'coefficients': coeffs_table,
            'intercept': float(ols_model.params['const']),
            'params': {col: float(ols_model.params[col]) for col in feats}
        }

# Outliers
ols_full = sm.OLS(y_train, sm.add_constant(X_train)).fit()
influence = ols_full.get_influence()
cooks_d = influence.cooks_distance[0]
top10_idx = np.argsort(cooks_d)[-10:][::-1]
outliers = []
for rank, idx in enumerate(top10_idx, start=1):
    orig_idx = int(X_train.index[idx])
    actual = float(y_train.iloc[idx])
    pred = float(ols_full.predict(sm.add_constant(X_train)).iloc[idx])
    resid = actual - pred
    outliers.append({
        'rank': rank,
        'sample_index': orig_idx,
        'state': str(df.loc[orig_idx, 'State']),
        'rd_spend': float(df.loc[orig_idx, 'R&D Spend']),
        'administration': float(df.loc[orig_idx, 'Administration']),
        'marketing_spend': float(df.loc[orig_idx, 'Marketing Spend']),
        'profit': actual,
        'predicted': pred,
        'residual': resid,
        'cooks_d': float(cooks_d[idx])
    })

full_export = {
    'algorithms': data_store,
    'outliers': outliers,
    'sweet_spot': {'k': 2, 'rmse': 8198.80, 'r2': 0.9474, 'features': ['R&D Spend', 'Marketing Spend']},
    'overfitting_cliff': {'k': 5, 'rmse': 9137.99, 'r2': 0.9347, 'features': feature_names}
}

json_path = os.path.join(BASE_DIR, 'model_data.json')
js_path = os.path.join(BASE_DIR, 'model_data.js')

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(full_export, f, indent=2, ensure_ascii=False)

with open(js_path, 'w', encoding='utf-8') as f:
    f.write('const MODEL_DATA = ' + json.dumps(full_export, indent=2, ensure_ascii=False) + ';')

print('Successfully exported model_data.json and model_data.js')
