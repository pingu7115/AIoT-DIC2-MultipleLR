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
    """Return the complete feature selection evaluation data."""
    dataset = request.args.get("dataset", "startups")
    if dataset in ALL_DATA:
        return jsonify(ALL_DATA[dataset])
    return jsonify(ALL_DATA)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
