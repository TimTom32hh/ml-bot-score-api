import pandas as pd 
import numpy as np
import pickle
import csv
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
import optuna
import os

from features import add_features, FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
REQUESTS_LOG =
LABELED_CSV =
MODEL_PATH   =

# ---------------------------------------------------------------------------
# 1) Roh-Log einlesen (nicht echte CSV)
# ---------------------------------------------------------------------------
try:
    with open(REQUESTS_LOG, "r", encoding="utf-8") as f:
        rows = []
        reader = csv.reader(f, delimiter=",", quotechar='"', skipinitialspace=True)
        for row in reader:
            if len(row) >= 7:
                rows.append(row)
    df_log = pd.DataFrame(rows)
except Exception as e:
    print(f"[Fehler] Datei nicht lesbar: {e}")
    exit(1)
if df_log.empty:
    print(f"[Fehler] '{REQUESTS_LOG}' enthält keine Daten.")
    exit(1)

# Spalten zuweisen
columns = [
    'timestamp', 'path', 'ua', 'blocked', 'blocked_reason',
    'status', 'country', 'cookie', 'method', 'referrer', 'fetchmode',
    'fetchsite', 'fetchdest', 'origin', 'language', 'encoding', 'connection',
    'accept', 'extra1', 'extra2'
]
df_log.columns = columns[:len(df_log.columns)]

# ---------------------------------------------------------------------------
# 2) Gelabelte CSV einlesen (mit Header, b/l am Ende)
# ---------------------------------------------------------------------------
try:
    df_labeled = pd.read_csv(LABELED_CSV, on_bad_lines='skip')
except Exception as e:
    print(f"[Fehler] Labeled CSV nicht lesbar: {e}")
    exit(1)

# Benennung anpassen
df_labeled.rename(columns={
    'Request': 'path',
    'User-Agent': 'ua',
    'Label': 'label'
}, inplace=True)

# b/l interpretieren
df_labeled['label'] = df_labeled['label'].astype(str).str.lower().str.strip()
df_labeled['target'] = df_labeled['label'].map({'b': 1, 'l': 0})
df_labeled = df_labeled[df_labeled['target'].isin([0, 1])]


# ---------------------------------------------------------------------------
# 3) Kombinieren (CSV hat Vorrang)
# ---------------------------------------------------------------------------
key_cols = ['path', 'ua']
df_log = df_log.dropna(subset=key_cols)
df_combined = pd.merge(df_log, df_labeled[key_cols + ['target']], on=key_cols, how='left')

# Zielspalte aus df_log verwenden, falls keine CSV-Zuordnung
if 'blocked' in df_combined.columns and 'target' in df_combined.columns:
    mask = df_combined['target'].isna()
    df_combined.loc[mask, 'blocked'] = df_combined.loc[mask, 'blocked'].astype(str).str.lower().str.strip()
    df_combined.loc[mask, 'target'] = df_combined.loc[mask, 'blocked'].map({
        '1': 1, 'true': 1, 'yes': 1, 'b': 1, 'blocked': 1,
        '0': 0, 'false': 0, 'no': 0, 'l': 0, 'allowed': 0
    })

# Nur gültige Labels behalten
df = df_combined[df_combined['target'].isin([0, 1])].copy()
if df.empty:
    print("[Fehler] Kein einziges gültiges Sample mit target-Wert.")
    exit(1)

# ---------------------------------------------------------------------------
# 4) Felder ergänzen
# ---------------------------------------------------------------------------
for col, default in [
    ('country', ''), ('method', 'GET'),
    ('allowed', ''), ('referrer', ''), ('MissingFetchHeaders', 0)
]:
    if col not in df.columns:
        df[col] = default

# ---------------------------------------------------------------------------
# 5) Feature-Engineering
# ---------------------------------------------------------------------------
df = add_features(df)

# ---------------------------------------------------------------------------
# 6) Matrix & Ziel
# ---------------------------------------------------------------------------
X = df[FEATURE_COLUMNS].astype(np.float32)
y = df['target']

# ---------------------------------------------------------------------------
# 7) Label-Verteilung prüfen
# ---------------------------------------------------------------------------
print("Label-Verteilung:", y.value_counts().to_dict())

# ---------------------------------------------------------------------------
# 8) Optuna-Tuning
# ---------------------------------------------------------------------------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'objective': 'binary',
        'is_unbalance': True,
        'random_state': 42
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for tr, val in skf.split(X, y):
        clf = LGBMClassifier(**params)
        clf.fit(X.iloc[tr], y.iloc[tr])
        proba = clf.predict_proba(X.iloc[val])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val], proba))
    return np.mean(aucs)

print("Optuna-Tuning …")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)
best_params = study.best_params
print("Beste Hyperparameter:", best_params)

# ---------------------------------------------------------------------------
# 9) Split & Training
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = LGBMClassifier(
    objective='binary', is_unbalance=True, random_state=42, **best_params
)
clf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 10) Calibration
# ---------------------------------------------------------------------------
min_cls = y_train.value_counts().min()
if min_cls < 2:
    calibrated_clf = clf
    print("Calibration übersprungen (zu wenige Samples).")
else:
    cv_folds = min(5, min_cls)
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=cv_folds)
    calibrated_clf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 11) Evaluation & Speichern
# ---------------------------------------------------------------------------
y_proba = calibrated_clf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
print("\nROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
print("Report:", classification_report(y_test, y_pred))

prec, rec, th = precision_recall_curve(y_test, y_proba)
f1 = 2 * prec * rec / (prec + rec + 1e-10)
best_thr = th[np.nanargmax(f1)] if len(th) > 0 else 0.5
print("Optimaler Threshold:", best_thr)

with open(MODEL_PATH, 'wb') as f:
    pickle.dump({
        'model': calibrated_clf,
        'optimal_threshold': float(best_thr),
        'feature_columns': FEATURE_COLUMNS
    }, f)

# ---------------------------------------------------------------------------
# 12) Feature Importances anzeigen
# ---------------------------------------------------------------------------
try:
    importances = calibrated_clf.estimator_.feature_importances_
except AttributeError:
    importances = clf.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n📊 Feature Importances:")
print(feature_importance_df.to_string(index=False))

print(f"✅ Modell gespeichert unter: {MODEL_PATH}")