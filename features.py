import numpy as np
import pandas as pd
import re

# --------------------------------------------------
# Regex-Patterns
# --------------------------------------------------
SENSITIVE_PATTERNS = re.compile(r'\.env|\.git|wp-login\.php|\.bash_history', re.IGNORECASE)
OUTDATED_PATTERN   = re.compile(r'Android\s+[12]\.|CPU iPhone OS (?:1[0-2]|12)_', re.IGNORECASE)
CRAWLER_PATTERN    = re.compile(r'bot|crawler|spider|slurp|archiver|bingpreview|nutch|yandex', re.IGNORECASE)
STRONG_CRAWLER_PATTERN = re.compile(r'python|scrapy|go-http-client|curl|java|wget', re.IGNORECASE)

# --------------------------------------------------
# Feature-Spalten
# --------------------------------------------------
FEATURE_COLUMNS = [
    "ua_length",
    "path_length",
    "ua_entropy",
    "path_entropy",
    "is_sensitive_path",
    "ua_outdated",
    "ua_crawler",
    "ua_crawler_strong",
    "ua_crawler_combined",
    "path_suspect_combined",
    "country_de",
    "country_other",
    "has_missing_headers"
]

# --------------------------------------------------
# Shannon-Entropie
# --------------------------------------------------
def shannon_entropy(s):
    if not isinstance(s, str) or len(s) == 0:
        return 0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum([p * np.log2(p) for p in prob if p > 0])

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
def add_features(df):
    df = df.copy()
    df['ua'] = df.get('ua', '').astype(str)
    df['path'] = df.get('path', '').astype(str)

    df['ua_length']    = df['ua'].str.len()
    df['path_length']  = df['path'].str.len()
    df['ua_entropy']   = df['ua'].apply(shannon_entropy)
    df['path_entropy'] = df['path'].apply(shannon_entropy)

    df['is_sensitive_path'] = df['path'].str.contains(SENSITIVE_PATTERNS, na=False).astype(int)
    df['ua_outdated']       = df['ua'].str.contains(OUTDATED_PATTERN, na=False).astype(int)
    df['ua_crawler']        = df['ua'].str.contains(CRAWLER_PATTERN, na=False).astype(int)
    df['ua_crawler_strong'] = df['ua'].str.contains(STRONG_CRAWLER_PATTERN, na=False).astype(int)

    # Kombinierte Features
    df['ua_crawler_combined'] = (
        (df['ua_crawler'] == 1) & (df['ua_entropy'] > 4)
    ).astype(int)

    df['path_suspect_combined'] = (
        (df['is_sensitive_path'] == 1) & (df['path_length'] > 30)
    ).astype(int)

    # Länderkennung
    df['country'] = df.get('country', '').astype(str).str.upper()
    df['country_de'] = (df['country'] == 'DE').astype(int)
    df['country_other'] = (df['country'] != 'DE').astype(int)

    # Header-Fetch-Checker
    df['MissingFetchHeaders'] = pd.to_numeric(df.get('MissingFetchHeaders', 0), errors='coerce').fillna(0).astype(int)
    df['has_missing_headers'] = (df['MissingFetchHeaders'] == 1).astype(int)

    return df
