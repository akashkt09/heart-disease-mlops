# app/main.py
"""Flask API serving the heart disease classification model."""

import pickle
import os

import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'heart_disease_pipeline.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

EXPECTED_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({'error': 'No JSON payload provided'}), 400

        missing = [f for f in EXPECTED_FEATURES if f not in data]
        if missing:
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        input_df = pd.DataFrame([{f: data[f] for f in EXPECTED_FEATURES}])

        prediction = int(model.predict(input_df)[0])
        confidence = float(model.predict_proba(input_df)[0][prediction])

        return jsonify({
            'prediction': prediction,
            'prediction_label': 'Disease Present' if prediction == 1 else 'No Disease',
            'confidence': round(confidence, 4)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)