# ml-bot-score-api

Ein leichtgewichtiger, selbst gehosteter ML-Service zur Erkennung von Bot‑Verkehr anhand technischer Merkmale in HTTP‑Requests.

## 🔍 Inhalte

- `serve_model.py` – Startet eine Flask‑API, die das Modell lädt und Anfragen bewertet
- `features.py` – Extrahiert und bereitet Request-Features zur Klassifikation auf
- `retrain_model.py` – Trainiert ein neues Modell auf Basis gelabelter Logs

## ⚙️ Nutzung

### 1. Setup lokal
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Server starten
```bash
python serve_model.py
```

### 3. Request testen
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp":"2025-06-24T08:00:00Z", "path":"/", "ua":"Mozilla/5.0", "method":"GET", "referrer":"", "status":200, "country":"DE", "MissingFetchHeaders":1}'
```

## 🔁 Nachtrainieren
```bash
python retrain_model.py
```
(Standardmäßig wird `requests.log` eingelesen und ein neues Modell gespeichert.)

## 🐳 Docker
```bash
docker build -t ml-bot-score-api .
docker run -p 5000:5000 --rm ml-bot-score-api
```

## 🛡️ Anwendungsbeispiel

Ideal geeignet als Bot-Erkennung:
- vor API-Endpunkten
- zur Formularabsicherung
- als Modul in Reverse Proxys (z. B. NGINX, HAProxy)
- DSGVO-konform, ohne Tracking oder externe Clouds

## 📄 Lizenz

MIT License – kommerzielle Nutzung erlaubt.

---

## 🤝 Kontakt

Für Fragen oder Kooperationen → [timtom32hh](https://github.com/timtom32hh)
