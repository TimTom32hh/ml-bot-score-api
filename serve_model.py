import pickle
import numpy as np
import pandas as pd
import os
from flask import Flask, request, jsonify

# Wir importieren add_features und FEATURE_COLUMNS aus features.py
from features import add_features, FEATURE_COLUMNS

MODEL_PATH = "/var/www/content-protector-release/ml-service/bot_model.pkl"

app = Flask(__name__)

# ---------------------------------------------------------------------
# 1. Modell laden
# ---------------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    payload = pickle.load(f)

# Falls payload ein Dict mit Encoder/Modell enthält, nehmen wir hier 'model'
clf = payload.get('model') if isinstance(payload, dict) else payload

# ---------------------------------------------------------------------
# 2. Inferenz‐Routine
# ---------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}

    # ---------------------------------------------------------------------
    # 2.1 – Default‐Spalten, die add_features() erwartet (wenn nicht übergeben)
    # ---------------------------------------------------------------------
    required = [
        'timestamp', 'path', 'ua', 'referrer', 'status',
        'country', 'method', 'MissingFetchHeaders',
        'blocked', 'allowed'
    ]
    for col in required:
        if col not in data:
            if col in ('status', 'MissingFetchHeaders'):
                data[col] = 0
            else:
                data[col] = ""

    # ---------------------------------------------------------------------
    # 2.2 – DataFrame mit genau einer Zeile erstellen
    # ---------------------------------------------------------------------
    df = pd.DataFrame([{
        'timestamp':           data['timestamp'],
        'path':                data['path'],
        'ua':                  data['ua'],
        'referrer':            data['referrer'],
        'status':              data['status'],
        'country':             data['country'],
        'method':              data['method'],
        'MissingFetchHeaders': data['MissingFetchHeaders'],
        'blocked':             data['blocked'],
        'allowed':             data['allowed']
    }])

    # ---------------------------------------------------------------------
    # 2.3 – Feature Engineering (genau wie im Training)
    # ---------------------------------------------------------------------
    df = add_features(df)

    # ---------------------------------------------------------------------
    # 2.4 – Nur die Spalten in FEATURE_COLUMNS extrahieren (als DataFrame, nicht np.array!)
    # ---------------------------------------------------------------------
    X = df[FEATURE_COLUMNS].astype(np.float32)

    # ---------------------------------------------------------------------
    # 2.5 – Vorhersage
    # ---------------------------------------------------------------------
    bot_score = float(clf.predict_proba(X)[0][1])
    return jsonify(bot_score=bot_score)

# ---------------------------------------------------------------------
# 3. Health-Check-Endpoint
# ---------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    model_name = type(clf).__name__
    return jsonify(status="OK", model=model_name)

if __name__ == '__main__':
    # Achtung: Nur Dev‐Server. Für Produktion besser Gunicorn/nginx o.Ä.
    app.run(host='xxx', port=0000)
