import os
import json
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates", static_folder=".")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DATA_PATH = os.path.join(BASE_DIR, "model_data.json")

try:
    with open(MODEL_DATA_PATH, "r", encoding="utf-8") as f:
        ALL_DATA = json.load(f)
except Exception as e:
    print(f"Warning: Could not load model_data.json: {e}")
    ALL_DATA = {}


@app.route("/")
def index():
    """Render the main StatsUnlock-inspired regression explorer dashboard."""
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return the complete 5-algorithm feature selection evaluation data."""
    dataset = request.args.get("dataset", "startups")
    algo = request.args.get("algo")
    
    ds_data = ALL_DATA.get(dataset, ALL_DATA.get("startups", {}))
    algos = ds_data.get("algorithms", {})
    if algo and algo in algos:
        return jsonify({algo: algos[algo]})
    return jsonify(algos if dataset in ALL_DATA else ALL_DATA)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Live prediction endpoint: Calculates predicted target based on inputs
    and the selected algorithm/k-feature subset.
    """
    try:
        data = request.get_json(force=True) or {}
        dataset = data.get("dataset", "startups")
        algo = data.get("algo", "Lasso")
        k = int(data.get("k", 2))

        ds_data = ALL_DATA.get(dataset, ALL_DATA.get("startups", {}))
        algos = ds_data.get("algorithms", {})
        if algo not in algos or k < 1 or k > len(algos[algo]):
            return jsonify({"error": "Invalid algorithm or k parameter"}), 400

        model_spec = algos[algo][k - 1]
        coefficients = model_spec["coefficients"]
        active_features = model_spec["features"]

        predicted = 0.0
        breakdown = []

        for c in coefficients:
            feat_name = c["feature"]
            coef_val = c["coef"]

            if "Intercept" in feat_name:
                predicted += coef_val
                breakdown.append({
                    "feature": "Intercept",
                    "value": 1.0,
                    "coef": coef_val,
                    "contribution": coef_val
                })
            elif feat_name in data:
                val = float(data[feat_name])
                contrib = coef_val * val
                predicted += contrib
                breakdown.append({
                    "feature": feat_name,
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "R&D Spend":
                val = float(data.get("rd_spend", 100000))
                contrib = coef_val * val
                predicted += contrib
                breakdown.append({
                    "feature": feat_name,
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "Marketing Spend":
                val = float(data.get("marketing_spend", 150000))
                contrib = coef_val * val
                predicted += contrib
                breakdown.append({
                    "feature": feat_name,
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "Administration":
                val = float(data.get("admin_spend", 120000))
                contrib = coef_val * val
                predicted += contrib
                breakdown.append({
                    "feature": feat_name,
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name.startswith("State_"):
                state = data.get("state", "California")
                cat_val = feat_name.split("_")[1]
                val = 1.0 if state == cat_val else 0.0
                contrib = coef_val * val
                predicted += contrib
                breakdown.append({
                    "feature": feat_name,
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })

        return jsonify({
            "status": "success",
            "dataset": dataset,
            "target": ds_data.get("target", "Target"),
            "algorithm": algo,
            "k": k,
            "active_features": active_features,
            "prediction": predicted,
            "rmse": model_spec["rmse"],
            "r2": model_spec["r2"],
            "breakdown": breakdown
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/summary", methods=["GET"])
def summary():
    """Return key comparison metrics across all 5 algorithms."""
    dataset = request.args.get("dataset", "startups")
    ds_data = ALL_DATA.get(dataset, ALL_DATA.get("startups", {}))
    algos = ds_data.get("algorithms", {})
    summary_data = []
    for algo, subsets in algos.items():
        for item in subsets:
            summary_data.append({
                "algorithm": algo,
                "k": item["k"],
                "features": item["features"],
                "rmse": item["rmse"],
                "r2": item["r2"],
                "mae": item["mae"]
            })
    return jsonify(summary_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f" * Starting Multiple Linear Regression Explorer on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
