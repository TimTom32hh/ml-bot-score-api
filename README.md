# ml-bot-score-api

Ein leichtgewichtiger ML-Service zur Erkennung von Bot‑Verkehr anhand verschiedener Request‑Features.

## Inhalte
- `serve_model.py` – Startet eine Flask-API, die `bot_model.pkl` lädt und ML‑Predictions liefert  
- `features.py` – Extraktionslogik für Request-Daten  
- `retrain_model.py` – Script zum Nachtrainieren und Export des Modells  

## Nutzung

1. Virtuelles Environment einrichten:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
