"""
fundamentals.py — Acceso a Financial Modeling Prep (FMP).

Consolida la lógica de los scripts de Clase 1:
  - Portfolio.py        → ratios TTM (P/E, P/B, EV/EBITDA, P/CF)
  - Estados_Contables.py → balance, income statement, cash flow
  - campos.py            → introspección de claves disponibles
"""

from __future__ import annotations

import os
import requests
import pandas as pd

FMP_BASE = "https://financialmodelingprep.com/stable"


def _get_fmp_key() -> str:
    """
    API key de FMP. Orden de prioridad:
      1. st.secrets["FMP_API_KEY"]   → para deploy (Streamlit Cloud / local secrets.toml)
      2. variable de entorno FMP_API_KEY
      3. ""  → sin key (el tab FUNDAMENTAL y la búsqueda FMP quedan inactivos)
    NUNCA se hardcodea la key en el código: así el repo puede ser público sin filtrarla.
    """
    try:
        import streamlit as st
        if "FMP_API_KEY" in st.secrets:
            return str(st.secrets["FMP_API_KEY"])
    except Exception:
        pass
    return os.environ.get("FMP_API_KEY", "")


FMP_API_KEY = _get_fmp_key()


# ─────────────────────────────────────────────
# RATIOS
# ─────────────────────────────────────────────
RATIOS = {
    "Firm Value / EBITDA": {
        "endpoint": "key-metrics-ttm",
        "candidates": [
            "evToEBITDATTM",
            "enterpriseValueOverEBITDATTM",
            "enterpriseValueMultipleTTM",
        ],
    },
    "Price / Earnings": {
        "endpoint": "ratios-ttm",
        "candidates": [
            "peRatioTTM",
            "priceToEarningsRatioTTM",
            "priceEarningsRatioTTM",
            "priceEarningsRatio",
        ],
    },
    "Price / Book Value": {
        "endpoint": "ratios-ttm",
        "candidates": [
            "pbRatioTTM",
            "priceToBookRatioTTM",
            "priceBookValueRatioTTM",
            "priceToBookRatio",
        ],
    },
    "Price / Cash Flow": {
        "endpoint": "ratios-ttm",
        "candidates": [
            "pocfratioTTM",
            "priceToOperatingCashFlowsRatioTTM",
            "priceToOperatingCashFlowRatioTTM",
            "priceCashFlowRatioTTM",
            "priceToCashFlowRatioTTM",
        ],
    },
}


def _fetch(endpoint: str, ticker: str, extra_params: dict | None = None) -> dict | list | None:
    url = f"{FMP_BASE}/{endpoint}"
    params = {"symbol": ticker, "apikey": FMP_API_KEY}
    if extra_params:
        params.update(extra_params)
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    data = r.json()
    if not data:
        return None
    return data


def _first_present(d: dict | None, candidates: list[str]):
    if not d:
        return None
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    return None


def get_ratios(tickers: list[str]) -> pd.DataFrame:
    """
    Devuelve un DataFrame con ratios en filas y tickers en columnas.
    """
    data = {name: [] for name in RATIOS}
    validos = []
    for ticker in tickers:
        cache: dict[str, dict | None] = {}
        for cfg in RATIOS.values():
            ep = cfg["endpoint"]
            if ep not in cache:
                resp = _fetch(ep, ticker)
                cache[ep] = resp[0] if isinstance(resp, list) and resp else None

        if all(v is None for v in cache.values()):
            continue

        for name, cfg in RATIOS.items():
            data[name].append(_first_present(cache.get(cfg["endpoint"]), cfg["candidates"]))
        validos.append(ticker)

    return pd.DataFrame(data, index=validos).T


# ─────────────────────────────────────────────
# ESTADOS CONTABLES
# ─────────────────────────────────────────────
STATEMENTS = {
    "Balance Sheet": "balance-sheet-statement",
    "Income Statement": "income-statement",
    "Cash Flow": "cash-flow-statement",
}


def get_statement(ticker: str, statement: str, period: str = "annual", limit: int = 5) -> pd.DataFrame | None:
    """
    statement: clave de STATEMENTS ('Balance Sheet', 'Income Statement', 'Cash Flow')
    period:    'annual' o 'quarter'
    """
    endpoint = STATEMENTS[statement]
    resp = _fetch(endpoint, ticker, {"period": period, "limit": limit})
    if not resp or not isinstance(resp, list):
        return None
    return pd.json_normalize(resp)


def get_multi_statement(tickers: list[str], statement: str, period: str = "annual") -> pd.DataFrame | None:
    """
    Devuelve un DataFrame consolidado (último período) con una fila por ticker.
    """
    rows = []
    for t in tickers:
        df = get_statement(t, statement, period=period, limit=1)
        if df is not None and not df.empty:
            df = df.copy()
            df.insert(0, "Ticker", t)
            rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def format_amount(x) -> str:
    """K/M/B formatting (igual que Estados_Contables.py)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.2f}M"
    if ax >= 1e3:
        return f"{x / 1e3:.2f}K"
    return f"{x:.2f}"
