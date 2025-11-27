"""
Feature engineering functions derived from your notebook:
- CharContinuationRate
- URLTitleMatchScore
- URLCharProb
- TLDLegitimateProb
"""

import numpy as np
import pandas as pd
import re
from collections import Counter
from urllib.parse import urlparse


def char_continuation_rate(url: str) -> float:
    if not isinstance(url, str) or len(url) < 2:
        return 0.0

    smooth = 0
    transitions = 0

    for c1, c2 in zip(url[:-1], url[1:]):
        transitions += 1
        if c1.isalnum() and c2.isalnum():
            smooth += 1

    return smooth / transitions if transitions else 0.0


def calculate_url_title_similarity(url: str, title: str) -> float:
    if not isinstance(url, str) or not isinstance(title, str):
        return 0.0

    url_tokens = re.split(r"[^A-Za-z0-9]+", url.lower())
    title_tokens = re.split(r"[^A-Za-z0-9]+", title.lower())

    url_tokens = [t for t in url_tokens if t]
    title_tokens = [t for t in title_tokens if t]

    if not url_tokens or not title_tokens:
        return 0.0

    url_count = Counter(url_tokens)
    title_count = Counter(title_tokens)

    overlap = sum(min(url_count[t], title_count[t]) for t in url_count if t in title_count)
    total = sum(url_count.values()) + sum(title_count.values())

    return (2 * overlap / total) if total else 0.0


def calculate_char_prob(url: str) -> float:
    if not isinstance(url, str):
        return 0.0

    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    score = 1.0

    for c in url.lower():
        if c in alphabet:
            score *= 0.03
        else:
            score *= 0.015

    return np.log(score) if score > 0 else -9999


TLD_SCORES = {
    "com": 0.9, "org": 0.85, "net": 0.8, "gov": 0.95,
    "edu": 0.95, "io": 0.6, "xyz": 0.4, "top": 0.3,
    "ru": 0.35, "cn": 0.35
}


def calculate_tld_legitimacy_score(url: str) -> float:
    try:
        tld = urlparse(url).netloc.split(".")[-1].lower()
        return TLD_SCORES.get(tld, 0.6)
    except:
        return 0.5


def add_all_features(df: pd.DataFrame, url_col="URL", title_col="Title"):
    df = df.copy()

    df["CharContinuationRate"] = df[url_col].apply(char_continuation_rate)

    if title_col in df.columns:
        df["URLTitleMatchScore"] = df.apply(
            lambda x: calculate_url_title_similarity(x[url_col], x[title_col]),
            axis=1,
        )
    else:
        df["URLTitleMatchScore"] = 0.0

    df["URLCharProb"] = df[url_col].apply(calculate_char_prob)
    df["TLDLegitimateProb"] = df[url_col].apply(calculate_tld_legitimacy_score)

    return df
