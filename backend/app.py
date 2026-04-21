import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymatgen.core import Structure
from werkzeug.utils import secure_filename

# Fix import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from router.router import route_crystal
from explainer.llm_explainer import generate_ai_explanation, answer_crystal_question
from models.predictor import predict as run_gnn_prediction

app = Flask(__name__)
CORS(app)

# Temp storage
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

# ✅ Health check (important for Render)
@app.route('/')
def home():
    return "API is running"

@app.route('/health')
def health():
    return {"status": "ok"}


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    cif_file = request.files['file']
    
    # ✅ safer filename
    filename = secure_filename(cif_file.filename)
    cif_path = os.path.join(TEMP_DIR, filename)
    cif_file.save(cif_path)

    try:
        structure = Structure.from_file(cif_path)
        pymatgen_formula = structure.composition.reduced_formula

        with open(cif_path, "r") as f:
            cif_text = f.read()

        # Model selection
        model_used = route_crystal(cif_path)

        # Prediction
        predictions = run_gnn_prediction(cif_path, model_used)

        return jsonify({
            **predictions,
            "model_used": model_used,
            "confidence": "High",
            "cif_text": cif_text,
            "formula": pymatgen_formula
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({
            "error": "Analysis failed. Check model weights and dependencies."
        }), 500

    finally:
        if os.path.exists(cif_path):
            os.remove(cif_path)


@app.route('/explain', methods=['POST'])
def explain():
    data = request.get_json()
    if not data or 'property' not in data or 'value' not in data:
        return jsonify({"error": "Missing property or value"}), 400

    try:
        explanation = generate_ai_explanation(data['property'], data['value'])
        return jsonify({"explanation": explanation})
    except Exception:
        return jsonify({"explanation": "Scientific review unavailable"}), 500


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data or 'predictions' not in data:
        return jsonify({"error": "Missing question or predictions"}), 400

    try:
        answer = answer_crystal_question(
            data['question'],
            data['predictions'],
            data.get('cif_text', '')
        )
        return jsonify({"answer": answer})
    except Exception:
        return jsonify({"answer": "Unable to answer at this time"}), 500


# ✅ Render-compatible entry point
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)