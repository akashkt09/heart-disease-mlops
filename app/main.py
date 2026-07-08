# app/main.py
"""Flask API serving the heart disease classification model."""

import pickle
import os
import time
import logging
from datetime import datetime, timezone
from collections import defaultdict

import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('api_requests.log'),
        logging.StreamHandler()  # also print to stdout, useful for `kubectl logs`
    ]
)
logger = logging.getLogger('heart-disease-api')

# --- In-memory metrics store ---
metrics = {
    'total_requests': 0,
    'prediction_counts': defaultdict(int),
    'total_response_time_ms': 0.0,
    'errors': 0,
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'heart_disease_pipeline.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

EXPECTED_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/metrics', methods=['GET'])
def get_metrics():
    avg_response_time = (
        metrics['total_response_time_ms'] / metrics['total_requests']
        if metrics['total_requests'] > 0 else 0
    )
    return jsonify({
        'total_requests': metrics['total_requests'],
        'errors': metrics['errors'],
        'prediction_distribution': dict(metrics['prediction_counts']),
        'average_response_time_ms': round(avg_response_time, 2)
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    metrics['total_requests'] += 1

    try:
        data = request.get_json()

        if data is None:
            metrics['errors'] += 1
            logger.warning("Request rejected: no JSON payload provided")
            return jsonify({'error': 'No JSON payload provided'}), 400

        missing = [f for f in EXPECTED_FEATURES if f not in data]
        if missing:
            metrics['errors'] += 1
            logger.warning(f"Request rejected: missing fields {missing}")
            return jsonify({'error': f'Missing required fields: {missing}'}), 400

        input_df = pd.DataFrame([{f: data[f] for f in EXPECTED_FEATURES}])

        prediction = int(model.predict(input_df)[0])
        confidence = float(model.predict_proba(input_df)[0][prediction])

        response_time_ms = (time.time() - start_time) * 1000
        metrics['prediction_counts'][str(prediction)] += 1
        metrics['total_response_time_ms'] += response_time_ms

        logger.info(
            f"Prediction served | age={data.get('age')} sex={data.get('sex')} "
            f"prediction={prediction} confidence={confidence:.4f} "
            f"response_time_ms={response_time_ms:.2f}"
        )

        return jsonify({
            'prediction': prediction,
            'prediction_label': 'Disease Present' if prediction == 1 else 'No Disease',
            'confidence': round(confidence, 4)
        }), 200

    except Exception as e:
        metrics['errors'] += 1
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)