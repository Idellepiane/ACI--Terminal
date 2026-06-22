"""
================================================================================
Bloomberg-style Dashboard · Administración de Carteras de Inversión I
--------------------------------------------------------------------------------
Tablero de control que integra todos los scripts del curso:
  - Clase 1:  Estados Contables + Portfolio (ratios)
  - Clase 2:  14 indicadores técnicos
  - Datos:    yahooquery (precios) + Financial Modeling Prep (fundamentales)

Cómo correrlo:
    pip install -r requirements.txt
    streamlit run app.py
================================================================================
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from yahooquery import Ticker

import fundamentals as fnd
import global_board as gb
import indicators as ind
import portfolio_theory as pt
import risk_attribution as ra
import strategies as stg
import universe as uni


# ════════════════════════════════════════════════════════════════════════════════
# CONFIG  · estilo Bloomberg (fondo negro, ámbar saturado, verde fluo, rojo)
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ACI I · Bloomberg Terminal",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLOOMBERG_CSS = """
<style>
    /* --- Tipografía --- */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inconsolata:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', 'Inconsolata', 'Menlo', 'Consolas', monospace !important;
        font-size: 13px;
    }

    /* --- Background base --- */
    .stApp { background-color: #000000; color: #d97a00; }
    .main .block-container { padding-top: 0.6rem; padding-bottom: 1rem; max-width: 100%; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); height: 0; }
    [data-testid="stToolbar"] { right: 8px; }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 2px solid #d97a00;
    }
    section[data-testid="stSidebar"] * { font-size: 12px !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #d97a00 !important; }

    /* --- Headers --- */
    h1, h2, h3, h4, h5 {
        color: #d97a00 !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.5px;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
    .stMarkdown, .stText, p, span, label, div { color: #c8c8c8; }

    /* --- Metric KPIs (Bloomberg-style: label arriba ámbar, valor blanco, delta color) --- */
    [data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #d97a00;
        padding: 6px 10px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #d97a00 !important;
        text-transform: uppercase;
        font-size: 0.65rem !important;
        letter-spacing: 1px;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem !important; font-weight: 700; }

    /* --- Buttons --- */
    .stButton button, .stDownloadButton button {
        background-color: #1a1a1a; color: #d97a00;
        border: 1px solid #d97a00; border-radius: 0;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase; font-weight: 700;
        font-size: 11px; padding: 4px 10px;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background-color: #d97a00; color: #000;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0a0a0a;
        border-bottom: 2px solid #d97a00;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a; color: #888;
        border-radius: 0;
        border-right: 1px solid #2a2a2a;
        padding: 6px 18px;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        font-size: 11px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #d97a00 !important;
        color: #000 !important;
    }

    /* --- DataFrames --- */
    .stDataFrame, .stTable { background-color: #0a0a0a !important; }
    div[data-testid="stDataFrame"] { border: 1px solid #2a2a2a; }
    div[data-testid="stDataFrame"] table { font-size: 11px !important; }
    div[data-testid="stDataFrame"] th {
        background-color: #1a1a1a !important;
        color: #d97a00 !important;
        text-transform: uppercase;
        font-weight: 700 !important;
        font-size: 10px !important;
        letter-spacing: 0.5px;
    }
    div[data-testid="stDataFrame"] td { color: #d8d8d8 !important; }

    /* --- Inputs (select, text_input, multiselect) --- */
    .stTextInput input, .stSelectbox > div, .stMultiSelect > div,
    .stNumberInput input {
        background-color: #1a1a1a !important;
        color: #d97a00 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-baseweb="select"] > div { background-color: #1a1a1a !important; }
    [data-baseweb="tag"] {
        background-color: #d97a00 !important;
        color: #000 !important;
        border-radius: 0 !important;
        font-weight: 700;
    }

    /* --- Bloomberg header bar (rojo con título y reloj) --- */
    .bbg-header {
        background: linear-gradient(180deg, #a00000 0%, #800000 100%);
        color: #ffffff;
        padding: 4px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 1px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #d97a00;
        margin: -0.6rem -1rem 0.5rem -1rem;
    }
    .bbg-header .left { display: flex; gap: 18px; align-items: center; }
    .bbg-header .right { display: flex; gap: 18px; align-items: center; }
    .bbg-shortcut { color: #e6a800; font-weight: 700; }
    .bbg-clock {
        color: #e6a800; font-weight: 700; font-size: 14px;
        background: #000; padding: 2px 10px; border: 1px solid #e6a800;
    }
    .bbg-market { color: #fff; text-transform: uppercase; }

    /* --- Ticker strip (estilo "Cash Market" panel) --- */
    .ticker-strip {
        background-color: #000;
        color: #d97a00;
        padding: 6px 12px;
        border: 1px solid #d97a00;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        overflow-x: auto;
        white-space: nowrap;
        margin-bottom: 6px;
    }
    .ticker-strip .sep { color: #444; padding: 0 8px; }

    /* --- Bloomberg panels (caja con borde y header ámbar) --- */
    .bbg-panel {
        background: #0a0a0a;
        border: 1px solid #2a2a2a;
        padding: 0;
        margin-bottom: 8px;
    }
    .bbg-panel-header {
        background: #1a1a1a;
        color: #d97a00;
        padding: 4px 10px;
        border-bottom: 1px solid #2a2a2a;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .bbg-panel-body { padding: 8px 10px; }

    /* --- Up/down colors (verde fluo y rojo saturado, Bloomberg) --- */
    .up   { color: #00ff41 !important; font-weight: 700; }
    .down { color: #ff2050 !important; font-weight: 700; }
    .flat { color: #888 !important; }
    .label { color: #d97a00; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; }
    .value { color: #ffffff; font-weight: 700; }

    hr { border-color: #2a2a2a; margin: 8px 0; }

    /* Reducir padding global para mayor densidad de info */
    .stPlotlyChart, .element-container { margin-bottom: 0.4rem; }
</style>
"""
st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)
st.markdown(gb.GLOBAL_CSS, unsafe_allow_html=True)


# Tema Plotly Bloomberg
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(family="JetBrains Mono, Menlo, monospace", color="#c8c8c8", size=10),
    xaxis=dict(
        gridcolor="#1a1a1a", zerolinecolor="#2a2a2a",
        linecolor="#d97a00", tickcolor="#d97a00",
        # Crosshair vertical que sigue al cursor (NO se imanta a un trace).
        showspikes=True, spikecolor="#d97a00", spikethickness=1,
        spikedash="dot", spikemode="across", spikesnap="cursor",
    ),
    yaxis=dict(
        gridcolor="#1a1a1a", zerolinecolor="#2a2a2a",
        linecolor="#d97a00", tickcolor="#d97a00",
        # Crosshair horizontal que sigue al cursor (NO se imanta a AAPL ni
        # al trace más cercano). Esto era el bug que veías en MARKET.
        showspikes=True, spikecolor="#d97a00", spikethickness=1,
        spikedash="dot", spikemode="across", spikesnap="cursor",
    ),
    legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor="#d97a00", borderwidth=1, font=dict(size=10)),
    margin=dict(l=40, r=20, t=20, b=25),
    # ── Navegación suave Bloomberg/TradingView ─────────────────────────────
    dragmode="pan",               # arrastrar = pan (no zoom-box)
    hovermode="x unified",        # tooltip único con todos los traces al hover x
    transition=dict(duration=0),  # sin lag entre updates
    uirevision="constant",        # preserva zoom/pan entre reruns/fragments
    # NOTA: NO se setea ni hoverdistance ni spikedistance. Default (20) es lo
    # correcto. spikedistance=-1 hace que el cursor "se imante" al trace más
    # cercano de toda la fig — eso era el bug del MARKET tab.
)

# Config interactivo de Plotly: scrollZoom + barra de herramientas limpia
PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "scrollZoom": True,                    # rueda del mouse = zoom suave
    "doubleClick": "reset",                # doble click = reset view
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d",
        "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian",
    ],
    "toImageButtonOptions": {
        "format": "png", "filename": "aci_chart",
        "height": 800, "width": 1400, "scale": 2,
    },
}

ACCENT = "#d97a00"     # Burnt amber — alto contraste con blanco sobre fondo negro


def _adj_price(df: pd.DataFrame) -> pd.Series:
    """
    Serie de precios para cálculos de retornos.

    Usa 'adjclose' (precio ajustado por dividendos y splits) cuando está
    disponible — necesario para CAPM, Markowitz y cualquier cálculo de
    retorno total. Si no hay adjclose (típico en intervalos intradía),
    cae a 'close' como fallback.
    """
    if "adjclose" in df.columns and df["adjclose"].notna().any():
        return df["adjclose"]
    return df["close"]
GREEN = "#00ff41"      # P/L verde fluo
RED = "#ff2050"        # P/L rojo saturado
CYAN = "#00d4ff"
PURPLE = "#bf5af2"
WHITE = "#ffffff"


# ════════════════════════════════════════════════════════════════════════════════
# PERSISTENCIA DE LA WATCHLIST  ·  memoria entre sesiones
# --------------------------------------------------------------------------------
# Streamlit olvida session_state al cerrar. Para que el dashboard recuerde la
# última watchlist al reabrirlo, la guardamos en un JSON local junto a app.py.
# Nota: en local es perfecto (sobrevive reinicios). En Streamlit Cloud el disco
# es efímero (se resetea al redeploy) y compartido entre visitantes, así que ahí
# la memoria es "best-effort": si no hay archivo, cae al default de abajo.
# ════════════════════════════════════════════════════════════════════════════════
_WATCHLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".watchlist.json")
_WATCHLIST_DEFAULT = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def _load_watchlist() -> list[str]:
    try:
        with open(_WATCHLIST_FILE, encoding="utf-8") as f:
            wl = json.load(f)
        if isinstance(wl, list):
            clean = [str(t).strip().upper() for t in wl if str(t).strip()]
            return clean if clean else _WATCHLIST_DEFAULT
    except Exception:
        pass
    return list(_WATCHLIST_DEFAULT)


def _save_watchlist(wl: list[str]) -> None:
    try:
        cur = list(wl)
        if cur == st.session_state.get("_wl_saved"):
            return  # evita reescribir el archivo en cada rerun si no cambió
        with open(_WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(cur, f)
        st.session_state["_wl_saved"] = cur
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════════
# DATA LAYER  · cacheado para no martillar APIs en cada refresh
# ════════════════════════════════════════════════════════════════════════════════
# Límites máximos de Yahoo Finance por intervalo (días hacia atrás).
# Si pedís más, Yahoo devuelve garbage y yahooquery crashea internamente.
_INTERVAL_MAX_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
    "1d": 36500,  # ~100 años, sin tope práctico
    "5d": 36500,
    "1wk": 36500,
    "1mo": 36500,
    "3mo": 36500,
}


def _clamp_range_for_interval(start: datetime, end: datetime, interval: str) -> tuple[datetime, str | None]:
    """
    Achica el rango si excede el máximo soportado por Yahoo para ese intervalo.
    Devuelve (start_efectivo, warning_msg | None).
    """
    max_days = _INTERVAL_MAX_DAYS.get(interval, 36500)
    span_days = (end - start).days
    if span_days > max_days:
        new_start = end - timedelta(days=max_days)
        msg = (f"Interval '{interval}' only supports up to {max_days} days of history "
               f"on Yahoo. Range trimmed: {new_start.date()} → {end.date()}.")
        return new_start, msg
    return start, None


def _normalize_index(idx):
    # yahooquery a veces devuelve fechas con timezones mezcladas (DST, etc.).
    # utc=True las unifica y tz_localize(None) deja el índice "naive" para Plotly.
    return pd.to_datetime(idx, utc=True, errors="coerce").tz_localize(None)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(tickers: tuple[str, ...], start: str, end: str, interval: str = "1d") -> dict[str, pd.DataFrame]:
    """
    Descarga OHLCV de cada ticker. Robusto: si el bulk-fetch falla (ej. un
    ticker corrupto), cae a un loop por ticker para que los demás funcionen.
    TTL 60s → cuasi-real-time.
    """
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    def _ingest(df: pd.DataFrame, tk_for_single: str):
        if df is None or df.empty:
            return
        if isinstance(df.index, pd.MultiIndex):
            for tk in df.index.get_level_values(0).unique():
                try:
                    sub = df.xs(tk, level=0).copy()
                    sub.index = _normalize_index(sub.index)
                    sub = sub[~sub.index.isna()]
                    if not sub.empty:
                        out[tk] = sub.sort_index()
                except (KeyError, Exception):
                    continue
        else:
            try:
                sub = df.copy()
                sub.index = _normalize_index(sub.index)
                sub = sub[~sub.index.isna()]
                if not sub.empty:
                    out[tk_for_single] = sub.sort_index()
            except Exception:
                pass

    # ── Intento 1: bulk fetch (más rápido) ─────────────────────────────────
    try:
        t = Ticker(list(tickers), asynchronous=True, validate=False)
        df = t.history(start=start, end=end, interval=interval)
        _ingest(df, tickers[0])
    except Exception:
        # yahooquery puede romperse internamente con un ticker corrupto.
        # No hacemos nada acá; el fallback de abajo cubre el resto.
        pass

    # ── Intento 2: por ticker faltante ─────────────────────────────────────
    # Si el bulk no trajo algún ticker, lo pedimos individualmente.
    for tk in tickers:
        if tk in out:
            continue
        try:
            sub_t = Ticker(tk, validate=False)
            df = sub_t.history(start=start, end=end, interval=interval)
            if isinstance(df, pd.DataFrame):
                _ingest(df, tk)
        except Exception:
            # Ticker que Yahoo no soporta para este interval/rango: lo skip-eamos.
            continue

    return out


@st.cache_data(ttl=30, show_spinner=False)
def fetch_quotes(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Snapshot live: precio, cambio, volumen, market cap.

    Primario: yahooquery .price (campos ricos; 1 request POR símbolo → usar
    sólo con listas chicas como la watchlist). Fallback: spark (batched, sin
    crumb) con columnas básicas si el crumb está caído o hay rate-limit.
    """
    if not tickers:
        return pd.DataFrame()

    rows = []
    try:
        # Mientras Yahoo nos tenga flaggeados (breaker activo), evitamos
        # también quoteSummary para no extender la penalización.
        if _yahoo_blocked():
            raise RuntimeError("yahoo rate-limited")
        t = Ticker(list(tickers))
        price = t.price
        for tk in tickers:
            p = price.get(tk, {}) if isinstance(price, dict) else {}
            if not isinstance(p, dict):
                continue
            # OJO: regularMarketChangePercent viene como FRACCIÓN (0.0152 = 1.52%).
            # Sin el ×100 todo el tablero muestra "0.02%" donde debería decir "1.52%".
            pct_raw = p.get("regularMarketChangePercent")
            rows.append({
                "Ticker": tk,
                "Nombre": p.get("longName") or p.get("shortName") or tk,
                "Último": p.get("regularMarketPrice"),
                "Cambio": p.get("regularMarketChange"),
                "Cambio %": pct_raw * 100 if isinstance(pct_raw, (int, float)) else None,
                "Apertura": p.get("regularMarketOpen"),
                "Máx Día": p.get("regularMarketDayHigh"),
                "Mín Día": p.get("regularMarketDayLow"),
                "Volumen": p.get("regularMarketVolume"),
                "Cap Mercado": p.get("marketCap"),
                "Divisa": p.get("currency", ""),
                "Mercado": p.get("exchangeName") or p.get("exchange", ""),
            })
    except Exception:
        rows = []

    df = pd.DataFrame(rows)
    if df.empty or df["Último"].notna().sum() == 0:
        # Fallback spark→chart: aproximamos apertura/máx/mín con closes intradía
        rows = []
        for sym, d in _market_data(tickers, rng="1d", interval="15m").items():
            closes, prev = d["closes"], d["prev"]
            last = closes[-1]
            rows.append({
                "Ticker": sym,
                "Nombre": sym,
                "Último": last,
                "Cambio": last - prev if prev else None,
                "Cambio %": (last / prev - 1) * 100 if prev else None,
                "Apertura": closes[0],
                "Máx Día": max(closes),
                "Mín Día": min(closes),
                "Volumen": None,
                "Cap Mercado": None,
                "Divisa": "",
                "Mercado": "YAHOO·SPARK",
            })
        df = pd.DataFrame(rows)
    return df


# Cadencia del price stream del tablero GLOBAL (segundos)
GB_POLL_SEC = 5


# ════════════════════════════════════════════════════════════════════════════════
# SPARK LAYER  ·  cotizaciones batched, sin crumb, resistente a rate-limits
# --------------------------------------------------------------------------------
# Lección aprendida: yahooquery .price pega a /v10/quoteSummary UNA request
# POR símbolo — con 122 tickers cada 2s eran ~60 req/s y Yahoo nos devolvió
# 429 + "Invalid Crumb" para toda la IP. El endpoint /v8/finance/spark acepta
# ~80 símbolos por request y NO usa crumb: todo el tablero entra en 2 requests
# por tick. Si aun así Yahoo tira 429, el circuit breaker frena los fetches
# por _SPARK_BACKOFF_SEC y la UI conserva los últimos valores en pantalla.
# ════════════════════════════════════════════════════════════════════════════════
_SPARK_URL = "https://query1.finance.yahoo.com/v8/finance/spark"
# Yahoo bajó el límite del batch: "Number of symbols needs to be less than or
# equal to 20" (HTTP 400 si pedís más). 87 símbolos Yahoo → 5 requests por tick.
_SPARK_CHUNK = 20
# Backoff LARGO ante 429: cuando Yahoo banea la IP, seguir poking cada 90s
# mantiene viva la penalización. Con 5 min de silencio inicial (creciendo a
# 30 min) le damos aire a la IP para que el ban decaiga solo.
_SPARK_BACKOFF_SEC = 300
_SPARK_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
_CHART_ROT_K = 6          # símbolos por tick en modo rotación (≈1.2 req/s)
_SPARK_BACKOFF_MAX = 1800  # backoff exponencial: 5 min → 10 → 20 → 30 min (tope)


@st.cache_resource(show_spinner=False)
def _spark_state() -> dict:
    # CLAVE: Yahoo aplica TLS fingerprinting — la huella de python-requests
    # recibe 429 sistemático (desde CUALQUIER IP), mientras que una huella de
    # Chrome real pasa. curl_cffi con impersonate="chrome" replica el TLS/JA3
    # y los headers de Chrome (no pisar el User-Agent: el perfil ya lo trae
    # coherente con la huella). Mismo fix que adoptó yfinance upstream.
    try:
        from curl_cffi import requests as _creq
        s = _creq.Session(impersonate="chrome")
    except ImportError:
        s = requests.Session()
        s.headers.update({"User-Agent": _SPARK_UA})
    # Breakers INDEPENDIENTES: spark (batch) y chart (per-símbolo) tienen
    # presupuestos de rate-limit distintos en Yahoo. Quedan como red de
    # seguridad ante rate-limits reales (de volumen), ya no deberían armarse
    # en operación normal.
    return {
        "session": s, "rot": 0,
        "spark_block": 0.0, "spark_backoff": float(_SPARK_BACKOFF_SEC),
        "chart_block": 0.0, "chart_backoff": float(_SPARK_BACKOFF_SEC),
    }


def _yahoo_blocked() -> bool:
    """True si AMBOS transportes Yahoo (spark y chart) están en penalty."""
    s = _spark_state()
    now = time.time()
    return now < s["spark_block"] and now < s["chart_block"]


def _spark_fetch(tickers: tuple[str, ...], rng: str = "1d", interval: str = "1d") -> dict[str, dict]:
    """
    Devuelve {sym: {'closes': [floats], 'prev': float|None}}.
    Batched (chunks de _SPARK_CHUNK símbolos), keep-alive. Ante un 429 arma
    backoff exponencial (90s → 15 min) y devuelve {} hasta que expire.
    """
    state = _spark_state()
    if time.time() < state["spark_block"]:
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(tickers), _SPARK_CHUNK):
        batch = tickers[i:i + _SPARK_CHUNK]
        try:
            r = state["session"].get(
                _SPARK_URL,
                params={"symbols": ",".join(batch), "range": rng, "interval": interval},
                timeout=8,
            )
        except Exception:
            continue
        if r.status_code == 429:
            state["spark_block"] = time.time() + state["spark_backoff"]
            state["spark_backoff"] = min(float(_SPARK_BACKOFF_MAX), state["spark_backoff"] * 2)
            break
        if not r.ok:
            continue
        try:
            js = r.json()
        except ValueError:
            continue
        if not isinstance(js, dict):
            continue
        state["spark_backoff"] = float(_SPARK_BACKOFF_SEC)  # respuesta sana → reset
        for sym, d in js.items():
            if not isinstance(d, dict):
                continue
            closes = [c for c in (d.get("close") or []) if c is not None]
            if not closes:
                continue
            out[sym] = {
                "closes": closes,
                "prev": d.get("previousClose") or d.get("chartPreviousClose"),
            }
    return out


def _chart_fetch(tickers: tuple[str, ...], rng: str = "1d", interval: str = "1d",
                 max_workers: int = 8) -> dict[str, dict]:
    """
    Fallback vía /v8/finance/chart (1 request POR símbolo, concurrencia acotada).
    Breaker propio: si acumula 429s (ráfaga), arma su backoff y no insiste. El
    poller lo usa de a 1 símbolo a 1.8s — ritmo que no dispara el límite.
    """
    state = _spark_state()
    if time.time() < state["chart_block"]:
        return {}

    def one(sym: str):
        try:
            r = state["session"].get(
                _CHART_URL.format(sym=requests.utils.quote(sym, safe="")),
                params={"range": rng, "interval": interval},
                timeout=8,
            )
            if r.status_code == 429:
                return sym, None, 429
            if not r.ok:
                return sym, None, r.status_code
            res = (r.json().get("chart") or {}).get("result")
            if not res:
                return sym, None, 200
            res = res[0]
            meta = res.get("meta") or {}
            quote = (res.get("indicators") or {}).get("quote") or [{}]
            closes = [c for c in (quote[0].get("close") or []) if c is not None] \
                if isinstance(quote[0], dict) else []
            last = meta.get("regularMarketPrice")
            if last is None and closes:
                last = closes[-1]
            if last is None:
                return sym, None, 200
            if not closes:
                closes = [last]
            prev = meta.get("chartPreviousClose") or meta.get("previousClose")
            return sym, {"closes": closes, "prev": prev}, 200
        except Exception:
            return sym, None, 0

    out: dict[str, dict] = {}
    n_429 = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sym, data, status in ex.map(one, tickers):
            if status == 429:
                n_429 += 1
            elif data:
                out[sym] = data
    if n_429 >= 3 or (n_429 and not out):
        state["chart_block"] = time.time() + state["chart_backoff"]
        state["chart_backoff"] = min(float(_SPARK_BACKOFF_MAX), state["chart_backoff"] * 2)
    elif out:
        state["chart_backoff"] = float(_SPARK_BACKOFF_SEC)
    return out


# Crypto vía Binance: gratis, sin crumb ni rate-limits prácticos. Mantiene
# vivo el panel CRYPTO (y la cinta) incluso cuando Yahoo banea la IP entera.
_BINANCE_URL = "https://api.binance.com/api/v3/ticker/24hr"
_BINANCE_MAP = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT",
    "BNB-USD": "BNBUSDT", "XRP-USD": "XRPUSDT", "ADA-USD": "ADAUSDT",
    "DOGE-USD": "DOGEUSDT", "AVAX-USD": "AVAXUSDT",
}


def _binance_fetch(tickers: tuple[str, ...]) -> dict[str, dict]:
    """{sym_yahoo: {'closes': [last], 'prev': close 24h}} para los crypto del tablero."""
    wanted = {y: b for y, b in _BINANCE_MAP.items() if y in tickers}
    if not wanted:
        return {}
    try:
        r = _spark_state()["session"].get(
            _BINANCE_URL,
            params={"symbols": json.dumps(list(wanted.values()), separators=(",", ":"))},
            timeout=8,
        )
        if not r.ok:
            return {}
        data = {d.get("symbol"): d for d in r.json() if isinstance(d, dict)}
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for ysym, bsym in wanted.items():
        d = data.get(bsym)
        if not d:
            continue
        try:
            last, prev = float(d["lastPrice"]), float(d["prevClosePrice"])
        except (KeyError, TypeError, ValueError):
            continue
        out[ysym] = {"closes": [last], "prev": prev}
    return out


# FX vía open.er-api.com: USD base, cobertura global (incl. ARS/CLP/PEN/UYU),
# gratis, sin key ni rate-limits. Mantiene vivo TODO el panel de monedas aunque
# Yahoo banee la IP. Da spot del día (sin prev-close → el %chg lo aporta Yahoo
# cuando está disponible; si no, queda neutro).
_ERAPI_URL = "https://open.er-api.com/v6/latest/USD"

# Cómo derivar cada par Yahoo desde rates[X] = X por 1 USD.
#   "inv:EUR"  → 1/rates[EUR]              (par moneda/USD, ej EUR/USD)
#   "dir:JPY"  → rates[JPY]                (par USD/moneda, ej USD/JPY)
#   "crs:JPY/EUR" → rates[JPY]/rates[EUR]  (cruce, ej EUR/JPY)
_FX_DERIVE = {
    "EURUSD=X": "inv:EUR", "GBPUSD=X": "inv:GBP", "AUDUSD=X": "inv:AUD", "NZDUSD=X": "inv:NZD",
    "USDJPY=X": "dir:JPY", "USDCHF=X": "dir:CHF", "USDCAD=X": "dir:CAD",
    "EURGBP=X": "crs:GBP/EUR", "EURJPY=X": "crs:JPY/EUR", "GBPJPY=X": "crs:JPY/GBP",
    "USDARS=X": "dir:ARS", "USDBRL=X": "dir:BRL", "USDMXN=X": "dir:MXN", "USDCLP=X": "dir:CLP",
    "USDCOP=X": "dir:COP", "USDPEN=X": "dir:PEN", "USDUYU=X": "dir:UYU", "USDCNY=X": "dir:CNY",
    "USDINR=X": "dir:INR", "USDKRW=X": "dir:KRW", "USDTRY=X": "dir:TRY", "USDZAR=X": "dir:ZAR",
}


@st.cache_data(ttl=300, show_spinner=False)
def _erapi_rates() -> dict[str, float]:
    try:
        r = _spark_state()["session"].get(_ERAPI_URL, timeout=8)
        if r.ok:
            js = r.json()
            if js.get("result") == "success":
                return js.get("rates", {})
    except Exception:
        pass
    return {}


def _erapi_fetch(tickers: tuple[str, ...]) -> dict[str, dict]:
    """{sym_yahoo: {'closes':[spot], 'prev':None}} para los pares FX del tablero."""
    wanted = [t for t in tickers if t in _FX_DERIVE]
    if not wanted:
        return {}
    rates = _erapi_rates()
    if not rates:
        return {}
    out: dict[str, dict] = {}
    for sym in wanted:
        rule = _FX_DERIVE[sym]
        try:
            mode, body = rule.split(":", 1)
            if mode == "dir":
                val = rates[body]
            elif mode == "inv":
                val = 1.0 / rates[body]
            else:  # crs:NUM/DEN
                num, den = body.split("/")
                val = rates[num] / rates[den]
        except (KeyError, ZeroDivisionError, ValueError):
            continue
        out[sym] = {"closes": [val], "prev": None}
    return out


def _market_data(tickers: tuple[str, ...], rng: str = "1d", interval: str = "1d") -> dict[str, dict]:
    """
    Transporte adaptativo multi-fuente (uso puntual: watchlist chica del header).
    spark → chart → relleno FX/crypto desde fuentes libres. El tablero GLOBAL
    NO usa esto: lee el poller en background (instantáneo).
    """
    out = _spark_fetch(tickers, rng, interval)
    if not out:
        out = _chart_fetch(tickers, rng, interval)
    for sym, d in {**_erapi_fetch(tickers), **_binance_fetch(tickers)}.items():
        out.setdefault(sym, d)
    return out


def _pack_quote(d: dict) -> dict:
    """{'closes':[...], 'prev':x} → {'last','prev','pct'}."""
    last = d["closes"][-1]
    prev = d.get("prev")
    return {"last": last, "prev": prev,
            "pct": (last / prev - 1) * 100 if prev else None}


# ════════════════════════════════════════════════════════════════════════════════
# BACKGROUND POLLER  ·  desacopla el fetcheo lento de la UI
# --------------------------------------------------------------------------------
# Un hilo daemon (uno por proceso) refresca un store compartido de cotizaciones
# a ritmo SUAVE: spark de todo el board cuando responde (1-2 requests), y si
# spark está baneado, rellena vía chart de a 1 símbolo cada ~1.7s (ritmo que se
# midió sostenible aun con la IP penalizada). FX (er-api) y crypto (Binance) se
# refrescan cada ciclo, gratis y sin límite. La UI sólo LEE el store → lecturas
# instantáneas, cero requests Yahoo en el camino del navegador, sin bans.
# ════════════════════════════════════════════════════════════════════════════════
_POLL_CHART_PER_CYCLE = 10      # símbolos Yahoo por ciclo cuando spark está caído
_POLL_CHART_GAP = 1.7           # segundos entre requests chart (ritmo seguro)


@st.cache_resource(show_spinner=False)
def _board_poller(yahoo_syms: tuple[str, ...], fx_syms: tuple[str, ...],
                  crypto_syms: tuple[str, ...]) -> dict:
    store: dict = {"quotes": {}, "lock": threading.Lock(), "yahoo_ok": False,
                   "cycles": 0, "started_at": time.time()}

    def _publish(snap: dict):
        if not snap:
            return
        with store["lock"]:
            store["quotes"].update(snap)

    def loop():
        rot = 0
        while True:
            try:
                # FX + crypto: batched, gratis, cada ciclo
                fxc = {}
                for sym, d in _erapi_fetch(fx_syms).items():
                    fxc[sym] = _pack_quote(d)
                for sym, d in _binance_fetch(crypto_syms).items():
                    fxc[sym] = _pack_quote(d)
                _publish(fxc)

                # Yahoo: intentar spark de TODO el board (1-2 requests).
                # _spark_fetch hace early-return SIN red si spark_block activo,
                # así que mientras dura el ban no tocamos Yahoo por acá.
                sp = _spark_fetch(yahoo_syms)
                if sp:
                    _publish({s: _pack_quote(d) for s, d in sp.items()})
                    with store["lock"]:
                        store["yahoo_ok"] = True
                        store["cycles"] += 1
                    time.sleep(GB_POLL_SEC)
                    continue

                with store["lock"]:
                    store["yahoo_ok"] = False

                # spark caído → rellenar vía chart de a poco, ritmo seguro.
                # Si el breaker de chart está activo (ban), NO tocamos Yahoo:
                # sólo servimos FX/crypto y esperamos a que el ban decaiga.
                if time.time() >= _spark_state()["chart_block"]:
                    for _ in range(_POLL_CHART_PER_CYCLE):
                        if time.time() < _spark_state()["chart_block"]:
                            break  # un 429 disparó el breaker → cortar la pasada
                        sym = yahoo_syms[rot % len(yahoo_syms)]
                        rot += 1
                        got = _chart_fetch((sym,), max_workers=1)
                        if got.get(sym):
                            _publish({sym: _pack_quote(got[sym])})
                        time.sleep(_POLL_CHART_GAP)
                with store["lock"]:
                    store["cycles"] += 1
                time.sleep(GB_POLL_SEC)
            except Exception:
                time.sleep(5)

    threading.Thread(target=loop, daemon=True, name="board-poller").start()
    return store


def _board_quotes(wait_first: float = 0.0) -> tuple[dict, bool]:
    """
    (quotes_snapshot, yahoo_ok) leído del poller. Instantáneo, sin red.
    wait_first>0: en el PRIMER render espera hasta ese tope a que el poller
    publique la primera tanda (FX+crypto, ~1-2s) para no pintar el tablero vacío.
    """
    crypto = tuple(s for s in gb.ALL_BOARD_TICKERS if s in _BINANCE_MAP)
    fx = tuple(s for s in gb.ALL_BOARD_TICKERS if s in _FX_DERIVE)
    yahoo = tuple(s for s in gb.ALL_BOARD_TICKERS if s not in _BINANCE_MAP and s not in _FX_DERIVE)
    store = _board_poller(yahoo, fx, crypto)
    deadline = time.time() + wait_first
    while wait_first > 0:
        with store["lock"]:
            if store["quotes"]:
                break
        if time.time() >= deadline:
            break
        time.sleep(0.2)
    with store["lock"]:
        return dict(store["quotes"]), store["yahoo_ok"]


def fetch_stream_quotes(tickers: tuple[str, ...]) -> dict[str, list]:
    """Snapshot {sym: [last, pct]} para el price stream — lee el poller."""
    q, _ = _board_quotes()
    return {sym: [v["last"], v["pct"]] for sym, v in q.items()}


def fetch_board_qmap(tickers: tuple[str, ...]) -> dict[str, dict]:
    """qmap del tablero GLOBAL: {sym: {last, chg, pct, name}} — lee el poller."""
    q, _ = _board_quotes(wait_first=3.0)
    return {
        sym: {
            "last": v["last"],
            "chg": (v["last"] - v["prev"]) if v.get("prev") else None,
            "pct": v["pct"],
            "name": sym,
        }
        for sym, v in q.items()
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_board_history(tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    """
    Cierres del último mes para los sparklines del tablero GLOBAL.
    Best-effort vía spark (1mo). Si Yahoo está baneado quedan sin sparkline
    (cosmético); el precio igual se ve por el poller.
    """
    out: dict[str, pd.Series] = {}
    for sym, d in _spark_fetch(tickers, rng="1mo", interval="1d").items():
        if len(d["closes"]) >= 2:
            out[sym] = pd.Series(d["closes"])
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_adjclose_matrix(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """
    Matriz ancha de precios ajustados (fecha × ticker) para los tabs RISK y LAB.
    Usa yahooquery .history (que sigue funcionando con la huella python — el
    ban de Yahoo afecta spark/chart/quoteSummary, no el endpoint de historia).
    """
    if not tickers:
        return pd.DataFrame()
    try:
        hist = Ticker(list(tickers), validate=False).history(start=start, end=end, interval="1d")
    except Exception:
        return pd.DataFrame()
    if not isinstance(hist, pd.DataFrame) or hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index()
    if "symbol" not in hist.columns or "adjclose" not in hist.columns:
        return pd.DataFrame()
    hist = hist[hist["symbol"].isin(tickers)]
    adj = hist.pivot(index="date", columns="symbol", values="adjclose")
    # Conservar solo los tickers pedidos que efectivamente vinieron, en orden
    cols = [t for t in tickers if t in adj.columns]
    return adj[cols].dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ratios(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fnd.get_ratios(list(tickers))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_statement(tickers: tuple[str, ...], statement: str, period: str) -> pd.DataFrame | None:
    return fnd.get_multi_statement(list(tickers), statement, period=period)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR  · controles
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ACI I TERMINAL")
    st.caption("v1.0")
    st.markdown("---")

    # ── Estado de la watchlist (persistente entre reruns Y entre sesiones) ──
    # Se guarda en un archivito local (.watchlist.json) para que, al reabrir el
    # dashboard, aparezca la última selección que dejaste — no el default fijo.
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = _load_watchlist()

    # ── 1) Selección rápida desde universos curados ─────────────────────────
    st.markdown("**UNIVERSE**")
    universe_group = st.selectbox(
        "Category",
        list(uni.UNIVERSE.keys()),
        index=0,
        label_visibility="collapsed",
    )
    picks_group = st.multiselect(
        f"Tickers in {universe_group}",
        options=uni.UNIVERSE[universe_group],
        default=[t for t in st.session_state.watchlist if t in uni.UNIVERSE[universe_group]],
        key=f"pick_{universe_group}",
        label_visibility="collapsed",
    )

    if st.button("ADD TO WATCHLIST", use_container_width=True):
        added = False
        for t in picks_group:
            if t not in st.session_state.watchlist:
                st.session_state.watchlist.append(t)
                added = True
        if added:
            st.rerun()

    # ── 2) Búsqueda dinámica (FMP, ~70k instrumentos globales) ─────────────
    with st.expander("SEARCH (FMP)", expanded=False):
        q = st.text_input("Symbol or name", value="", placeholder="Berkshire, GGAL, EURUSD")
        if q:
            res = uni.search_fmp(q, limit=20)
            if not res.empty:
                res_display = res.copy()
                if "symbol" in res_display.columns and "name" in res_display.columns:
                    res_display["label"] = res_display["symbol"] + " — " + res_display["name"].fillna("")
                    pick = st.selectbox("Results", options=res_display["label"].tolist(), index=0)
                    chosen_symbol = pick.split(" — ")[0]
                    if st.button(f"ADD {chosen_symbol}", use_container_width=True):
                        if chosen_symbol not in st.session_state.watchlist:
                            st.session_state.watchlist.append(chosen_symbol)
                            st.rerun()
            else:
                st.caption("No results.")

    # ── 3) Ticker manual ────────────────────────────────────────────────────
    with st.expander("MANUAL ENTRY", expanded=False):
        manual = st.text_input("Symbol (e.g. GGAL.BA, BTC-USD)", value="").strip().upper()
        if manual and st.button("ADD", use_container_width=True, key="add_manual"):
            if manual not in st.session_state.watchlist:
                st.session_state.watchlist.append(manual)
                st.rerun()

    # ── Watchlist actual ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**WATCHLIST**")
    st.session_state.watchlist = st.multiselect(
        "Active tickers",
        options=sorted(set(st.session_state.watchlist + uni.all_tickers())),
        default=st.session_state.watchlist,
        label_visibility="collapsed",
    )
    if st.button("CLEAR", use_container_width=True):
        st.session_state.watchlist = []
        _save_watchlist([])
        st.rerun()

    # Persistimos la selección actual para la próxima vez que abras el dashboard.
    _save_watchlist(st.session_state.watchlist)

    tickers = tuple(t.strip().upper() for t in st.session_state.watchlist if t.strip())

    st.markdown("---")
    st.markdown("**PRIMARY**")
    primary = st.selectbox(
        "Primary ticker",
        options=tickers if tickers else ["AAPL"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**PERÍODO DE ESTUDIO**")
    st.caption("Aplica a TODOS los tabs: Market · Technical · Theory · Portfolio · Risk · Strategy Lab.")

    end_date = datetime.now()
    # Presets estilo terminal. Días hacia atrás (YTD se calcula aparte; MÁX = ~30 años).
    _PRESETS = {
        "1M": 30, "6M": 182, "YTD": None, "1A": 365, "2A": 730,
        "5A": 365 * 5, "10A": 365 * 10, "MÁX": 365 * 30,
    }
    period_choice = st.segmented_control(
        "Período", list(_PRESETS.keys()), default="5A",
        label_visibility="collapsed", key="period_seg",
    ) or "5A"

    if period_choice == "YTD":
        start_date = datetime(end_date.year, 1, 1)
    else:
        start_date = end_date - timedelta(days=_PRESETS[period_choice])

    # Rango a medida + intervalo de velas (plegado para no saturar)
    with st.expander("Rango a medida · intervalo de velas", expanded=False):
        use_custom = st.checkbox("Usar fechas a medida", value=False, key="use_custom_range")
        if use_custom:
            cda, cdb = st.columns(2)
            with cda:
                _sd = st.date_input("Desde", value=start_date.date(),
                                    min_value=datetime(1970, 1, 1).date(),
                                    max_value=end_date.date(), key="cr_start")
            with cdb:
                _ed = st.date_input("Hasta", value=end_date.date(),
                                    min_value=datetime(1970, 1, 1).date(),
                                    max_value=end_date.date(), key="cr_end")
            start_date = datetime.combine(_sd, datetime.min.time())
            end_date = datetime.combine(_ed, datetime.min.time())
            if start_date >= end_date:
                st.error("La fecha de inicio debe ser anterior a la de fin.")
        interval_choice = st.selectbox(
            "Intervalo de velas",
            ["1m", "5m", "15m", "30m", "1h", "1d", "1wk"], index=5,
            help="Los intervalos intradía (1m/5m/...) solo cargan los últimos días por límites de Yahoo.",
            key="interval_sel",
        )

    st.caption(f"Datos del **{start_date.date()}** al **{end_date.date()}** · intervalo **{interval_choice}**")

    st.markdown("---")
    st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(f"Data: Yahoo Finance · FMP · live stream {GB_POLL_SEC}s")


# ════════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════════
if not tickers:
    st.error("Add at least one ticker in the sidebar.")
    st.stop()

# Validamos que el intervalo+rango sea soportado por Yahoo Finance.
# Si no, achicamos el rango automáticamente y mostramos el aviso.
start_date, range_warning = _clamp_range_for_interval(start_date, end_date, interval_choice)
if range_warning:
    st.warning(range_warning)

with st.spinner("Loading market data..."):
    history = fetch_history(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval_choice,
    )
    quotes = fetch_quotes(tickers)

# Reportamos tickers que no se pudieron cargar (para que el usuario sepa)
missing = [t for t in tickers if t not in history]
if missing and len(missing) < len(tickers):
    st.caption(f"Note: no data returned for {', '.join(missing)} "
               f"(unsupported by Yahoo for interval={interval_choice}).")


# ════════════════════════════════════════════════════════════════════════════════
# BLOOMBERG HEADER + TICKER STRIP + KPIs  ·  EN FRAGMENT (refresh parcial)
# --------------------------------------------------------------------------------
# Refresca KPIs (OPEN/HIGH/LOW/VOL/MKT CAP) cada 30s en silencio. Los PRECIOS
# del strip los pisa el price stream del GLOBAL cada 2s (spans data-sym), y el
# reloj tickea client-side cada 1s. El rerun del fragment NO toca el resto de
# la app: charts, tabs activos, scroll y dropdowns quedan como están.
# ════════════════════════════════════════════════════════════════════════════════
# Refresh interno del header (KPIs OPEN/HIGH/LOW/VOL que el price stream no
# cubre). Siempre activo, sin control de usuario: los precios ya laten cada
# GB_POLL_SEC segundos vía el stream del GLOBAL.
_live_every = 30


@st.fragment(run_every=_live_every)
def live_header():
    # Reloj y header bar
    now_str = datetime.now().strftime("%H:%M:%S")
    market_label = f"{primary} · GLOBAL MARKETS" if primary else "GLOBAL MARKETS"
    header_html = f"""
    <div class="bbg-header">
      <div class="left">
        <span class="bbg-shortcut">97)</span><span>Regions</span>
        <span class="bbg-shortcut">98)</span><span>Settings</span>
        <span class="bbg-shortcut">99)</span><span>Help</span>
      </div>
      <div class="right">
        <span class="bbg-clock">{now_str}</span>
        <span class="bbg-market">{market_label}</span>
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Re-pedimos cotizaciones (cacheadas a 30s) y reconstruimos el strip + KPIs.
    # Importante: si quotes pesa, el cache evita el round-trip de red.
    live_quotes = fetch_quotes(tickers)

    if not live_quotes.empty:
        rows_html = []
        for _, r in live_quotes.iterrows():
            if pd.isna(r["Último"]):
                continue
            # data-sym → el price stream del GLOBAL pisa estos spans en vivo
            rows_html.append(
                f"<b style='color:#d97a00'>{r['Ticker']}</b>&nbsp;"
                f"<span class='value gb-last' data-sym='{r['Ticker']}'>{r['Último']:.2f}</span>&nbsp;"
                f"<span class='gb-pct' data-sym='{r['Ticker']}'>{gb._chg_span(r['Cambio %'])}</span>"
            )
        strip_html = "<span class='sep'>|</span>".join(rows_html)
        st.markdown(f"<div class='ticker-strip'>{strip_html}</div>", unsafe_allow_html=True)

    if not live_quotes.empty and primary in live_quotes["Ticker"].values:
        q = live_quotes[live_quotes["Ticker"] == primary].iloc[0]
        name = q.get("Nombre", primary)
        st.markdown(
            f"<h2 style='margin:8px 0 4px 0'>{primary} "
            f"<span style='color:#888;font-weight:400;font-size:0.7em'>· {name}</span></h2>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("LAST", f"{q['Último']:.2f}" if pd.notna(q['Último']) else "—",
                  delta=f"{q['Cambio %']:+.2f}%" if pd.notna(q['Cambio %']) else None)
        c2.metric("OPEN", f"{q['Apertura']:.2f}" if pd.notna(q['Apertura']) else "—")
        c3.metric("HIGH", f"{q['Máx Día']:.2f}" if pd.notna(q['Máx Día']) else "—")
        c4.metric("LOW", f"{q['Mín Día']:.2f}" if pd.notna(q['Mín Día']) else "—")
        c5.metric("VOLUME", fnd.format_amount(q['Volumen']))
        c6.metric("MKT CAP", fnd.format_amount(q['Cap Mercado']))


live_header()


# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
(tab_global, tab_market, tab_tech, tab_fund, tab_theory, tab_port,
 tab_risk, tab_lab) = st.tabs(
    ["GLOBAL", "MARKET", "TECHNICAL", "FUNDAMENTAL", "THEORY", "PORTFOLIO",
     "RISK", "STRATEGY LAB"]
)


@st.cache_data(ttl=3600, show_spinner=False)
def run_ml_backtest(tickers: tuple[str, ...], start: str, end: str,
                    model_choice: str, window: int, cost: float):
    """Walk-forward ML (timing on/off) cacheado — reentrena ~miles de modelos."""
    adj = fetch_adjclose_matrix(tickers, start, end)
    if adj.empty or adj.shape[1] == 0:
        return None
    w = np.array([1 / adj.shape[1]] * adj.shape[1])
    return stg.backtest_ml(adj, w, model_choice=model_choice, window=window, cost=cost)


@st.cache_data(ttl=3600, show_spinner=False)
def run_ml_weights_backtest(tickers: tuple[str, ...], start: str, end: str,
                            model_choice: str, window: int, cost: float,
                            umbral: float, max_peso: float | None):
    """Walk-forward ML con pesos dinámicos (un modelo por activo) cacheado."""
    adj = fetch_adjclose_matrix(tickers, start, end)
    if adj.empty or adj.shape[1] < 2:
        return None
    return stg.backtest_ml_weights(adj, model_choice=model_choice, window=window,
                                   cost=cost, umbral=umbral, max_peso=max_peso)


# ──────────────────────────────────────────────────────────────────────────────
# TAB · GLOBAL  ·  tablero mundial estilo Bloomberg WEI/GMM
# --------------------------------------------------------------------------------
# Independiente de la watchlist: muestra SIEMPRE el mundo entero (índices por
# región, FX, commodities, tasas, vol, crypto y panel Argentina con CCL).
#
# Arquitectura: un POLLER en segundo plano (hilo daemon, ver _board_poller)
# refresca un store de cotizaciones a ritmo suave desde múltiples fuentes
# (Yahoo spark/chart + er-api FX + Binance crypto). La UI sólo LEE ese store
# (instantáneo). El tablero se dibuja con TODAS las filas siempre (placeholder
# "···" donde aún no hay dato); el price stream (fragment invisible, cada
# GB_POLL_SEC) pisa los <span data-sym> con flash verde/rojo, y un rerun
# periódico refresca movers/treemap/curva/sparklines a medida que el poller
# completa el tablero. Nunca queda inoperativa ni depende de un solo feed.
# ──────────────────────────────────────────────────────────────────────────────
with tab_global:
    qmap = fetch_board_qmap(gb.ALL_BOARD_TICKERS)
    _, _yahoo_ok = _board_quotes()
    if qmap:
        st.session_state["_gb_qmap"] = qmap          # último snapshot bueno
    qmap = {**st.session_state.get("_gb_qmap", {}), **qmap}
    sparks = fetch_board_history(gb.ALL_BOARD_TICKERS)

    n_have = len(qmap)
    n_total = len(gb.ALL_BOARD_TICKERS)

    if n_have == 0:
        st.info("Inicializando el tablero — bajando cotizaciones…")
    elif n_have < n_total and not _yahoo_ok:
        st.caption(f"📡 {n_have}/{n_total} live · FX y crypto en tiempo real. "
                   f"Índices, commodities y acciones (Yahoo) están temporalmente rate-limiteados — "
                   f"se completan solos en cuanto Yahoo libera la IP (puede tardar unos minutos).")
    elif n_have < n_total:
        st.caption(f"⏳ Cargando tablero — {n_have}/{n_total} instrumentos listos…")

    # Cinta animada + relojes de plazas mundiales
    st.markdown(gb.tape_html(qmap), unsafe_allow_html=True)
    st.markdown(gb.sessions_html(), unsafe_allow_html=True)

    # ── Índices bursátiles por región ────────────────────────────────
    cols = st.columns(3)
    for col, (region, rows) in zip(cols, gb.EQUITY_REGIONS.items()):
        with col:
            st.markdown(gb.panel_html(region, rows, qmap, sparks), unsafe_allow_html=True)

    # ── FX y crypto ──────────────────────────────────────────────────
    cols = st.columns(3)
    for col, (grp, rows) in zip(cols, gb.FX_PANELS.items()):
        with col:
            st.markdown(gb.panel_html(grp, rows, qmap, sparks), unsafe_allow_html=True)

    # ── Commodities ──────────────────────────────────────────────────
    cols = st.columns(3)
    for col, (grp, rows) in zip(cols, gb.COMMODITY_PANELS.items()):
        with col:
            st.markdown(gb.panel_html(grp, rows, qmap, sparks), unsafe_allow_html=True)

    # ── Tasas US + volatilidad + Argentina ───────────────────────────
    c_rates, c_vol, c_arg = st.columns([1.3, 1, 1.4])
    with c_rates:
        st.markdown("<div class='bbg-panel'><div class='bbg-panel-header'>"
                    "CURVA TREASURIES US</div></div>", unsafe_allow_html=True)
        fig_curve = gb.curve_fig(qmap, PLOTLY_LAYOUT)
        if fig_curve is not None:
            st.plotly_chart(fig_curve, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.caption("Curva de tasas: esperando datos de Yahoo…")
    with c_vol:
        st.markdown(gb.panel_html("VOLATILIDAD", gb.VOL_PANEL, qmap, sparks),
                    unsafe_allow_html=True)
        st.markdown(gb.panel_html("TASAS US (NIVEL %)",
                                  [(s, n, "YIELD") for s, n, _ in gb.RATES],
                                  qmap, sparks),
                    unsafe_allow_html=True)
    with c_arg:
        st.markdown(gb.ars_summary_html(qmap), unsafe_allow_html=True)
        st.markdown(gb.panel_html("ARGENTINA · ADRs NYSE", gb.ARG_ADRS, qmap, sparks),
                    unsafe_allow_html=True)

    # ── Movers + treemap · en fragment que relee el poller cada 15s ──
    # Las FILAS de los paneles ya se llenan solas vía el price stream (tienen
    # ancla data-sym aunque arranquen en "···"). Movers/treemap son estáticos
    # y sí necesitan re-render para reordenarse a medida que entran símbolos:
    # este fragment lo hace LOCALMENTE (scope fragment) → sin recargar la
    # página, sin reset de scroll, sin spinner global.
    @st.fragment(run_every=15)
    def gb_movers_treemap():
        q_now, _ = _board_quotes()
        q_now = {**st.session_state.get("_gb_qmap", {}), **{
            s: {"last": v["last"], "chg": (v["last"] - v["prev"]) if v.get("prev") else None,
                "pct": v["pct"], "name": s} for s, v in q_now.items()}}
        g_html, l_html = gb.movers_html(q_now)
        c_g, c_l = st.columns(2)
        c_g.markdown(g_html, unsafe_allow_html=True)
        c_l.markdown(l_html, unsafe_allow_html=True)
        fig_map = gb.treemap_fig(q_now, "JetBrains Mono, Menlo, monospace")
        if fig_map is not None:
            st.plotly_chart(fig_map, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(f"GLOBAL BOARD · {len(q_now)}/{n_total} instrumentos · "
                   f"fuentes: Yahoo (índices/commodities/acciones) · er-api (FX) · Binance (crypto) · "
                   f"price stream {GB_POLL_SEC}s · sparklines = último mes")

    gb_movers_treemap()

    # ── PRICE STREAM · latido Bloomberg, siempre activo ──────────────
    # Fragment invisible: lee el poller y pisa los <span data-sym> del DOM,
    # rellenando también las filas que arrancaron en "···".
    _stream_tickers = tuple(dict.fromkeys(gb.ALL_BOARD_TICKERS + tickers))

    @st.fragment(run_every=GB_POLL_SEC)
    def gb_price_stream():
        payload = fetch_stream_quotes(_stream_tickers)
        if payload:
            components.html(
                gb.stream_js(payload, salt=datetime.now().strftime("%H%M%S%f")),
                height=0,
            )

    gb_price_stream()


# ──────────────────────────────────────────────────────────────────────────────
# TAB · MERCADO
# ──────────────────────────────────────────────────────────────────────────────
with tab_market:
    st.markdown("##### QUOTES")

    if quotes.empty:
        st.warning("Could not fetch quotes.")
    else:
        styled = quotes.copy()
        for col in ["Último", "Cambio", "Apertura", "Máx Día", "Mín Día"]:
            styled[col] = styled[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        styled["Cambio %"] = styled["Cambio %"].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        styled["Volumen"] = styled["Volumen"].apply(fnd.format_amount)
        styled["Cap Mercado"] = styled["Cap Mercado"].apply(fnd.format_amount)
        st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### COMPARATIVE PERFORMANCE  ·  BASE 100")

    if history:
        fig = go.Figure()
        for tk, df in history.items():
            if df.empty or "close" not in df.columns:
                continue
            norm = df["close"] / df["close"].iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=df.index, y=norm, mode="lines", name=tk,
                line=dict(width=1.6),
                # hovertemplate explícito → en hover unificado cada ticker aparece
                # con su valor; sin esto Plotly usa default y se confunde con
                # múltiples traces que se superponen.
                hovertemplate="<b>%{fullData.name}</b>  %{y:.2f}<extra></extra>",
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    st.markdown("##### DAILY HEATMAP  ·  LAST SESSION %CHG")

    if not quotes.empty:
        hm = quotes[["Ticker", "Cambio %"]].dropna().sort_values("Cambio %", ascending=False)
        if not hm.empty:
            fig_hm = go.Figure(go.Bar(
                x=hm["Ticker"], y=hm["Cambio %"],
                marker=dict(color=[GREEN if v > 0 else RED for v in hm["Cambio %"]]),
                text=[f"{v:+.2f}%" for v in hm["Cambio %"]], textposition="outside",
            ))
            fig_hm.update_layout(**PLOTLY_LAYOUT, height=350, yaxis_title="%CHG", showlegend=False)
            st.plotly_chart(fig_hm, use_container_width=True, config=PLOTLY_CONFIG)


# ──────────────────────────────────────────────────────────────────────────────
# TAB · ANÁLISIS TÉCNICO
# ──────────────────────────────────────────────────────────────────────────────
with tab_tech:
    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.markdown("##### INDICATOR")
        indicator_name = st.selectbox(
            "Indicator",
            list(ind.INDICATORS.keys()),
            index=0,
            label_visibility="collapsed",
        )
        cfg = ind.INDICATORS[indicator_name]
        st.caption(f"Type: {cfg['kind'].upper()}")

        st.markdown("**PARAMETERS**")
        params = {}
        for k, default in cfg["params"].items():
            if isinstance(default, bool):
                params[k] = st.checkbox(k, value=default)
            elif isinstance(default, int):
                params[k] = st.number_input(k, value=default, step=1)
            elif isinstance(default, float):
                params[k] = st.number_input(k, value=float(default), step=0.5, format="%.2f")
            else:
                params[k] = st.text_input(k, value=str(default))

    with col_b:
        st.markdown(f"##### {primary}  ·  {indicator_name.upper()}")

        df = history.get(primary)
        if df is None or df.empty:
            st.warning(f"No data for {primary}.")
        else:
            kind = cfg["kind"]

            if kind == "overlay":
                # Superpuesto al precio (Bollinger, Keltner, LR-bands)
                d = cfg["fn"](df, **params)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                    name="OHLC", increasing_line_color=GREEN, decreasing_line_color=RED,
                ))
                if indicator_name == "Bandas de Bollinger":
                    fig.add_trace(go.Scatter(x=d.index, y=d["MA"], name="MA20", line=dict(color=ACCENT)))
                    fig.add_trace(go.Scatter(x=d.index, y=d["Upper"], name="Upper", line=dict(color=CYAN, dash="dash")))
                    fig.add_trace(go.Scatter(x=d.index, y=d["Lower"], name="Lower", line=dict(color=CYAN, dash="dash"),
                                              fill="tonexty", fillcolor="rgba(0,212,255,0.05)"))
                elif indicator_name == "Bandas de Keltner":
                    fig.add_trace(go.Scatter(x=d.index, y=d["KC_Central"], name="Central", line=dict(color=ACCENT)))
                    fig.add_trace(go.Scatter(x=d.index, y=d["KC_Upper"], name="Upper", line=dict(color=GREEN, dash="dash")))
                    fig.add_trace(go.Scatter(x=d.index, y=d["KC_Lower"], name="Lower", line=dict(color=RED, dash="dash"),
                                              fill="tonexty", fillcolor="rgba(255,255,255,0.03)"))
                elif indicator_name == "Bandas de Auto-Regresión":
                    fig.add_trace(go.Scatter(x=d.index, y=d["LR_Recta"], name="Tendencia", line=dict(color=ACCENT, width=2)))
                    fig.add_trace(go.Scatter(x=d.index, y=d["LR_Upper"], name=f"+{params.get('k', 2)}σ", line=dict(color=GREEN, dash="dash")))
                    fig.add_trace(go.Scatter(x=d.index, y=d["LR_Lower"], name=f"-{params.get('k', 2)}σ", line=dict(color=RED, dash="dash"),
                                              fill="tonexty", fillcolor="rgba(255,255,255,0.03)"))
                fig.update_layout(**PLOTLY_LAYOUT, height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            elif kind == "oscillator":
                # Panel doble: precio arriba, oscilador abajo
                d = cfg["fn"](df, **params)
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.4], vertical_spacing=0.04,
                )
                fig.add_trace(go.Candlestick(
                    x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                    name="OHLC", increasing_line_color=GREEN, decreasing_line_color=RED,
                ), row=1, col=1)

                if indicator_name == "RSI (Wilder)":
                    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI", line=dict(color=PURPLE)), row=2, col=1)
                    fig.add_hline(y=70, line=dict(color=RED, dash="dash"), row=2, col=1)
                    fig.add_hline(y=30, line=dict(color=GREEN, dash="dash"), row=2, col=1)
                    fig.update_yaxes(range=[0, 100], row=2, col=1)

                elif indicator_name == "MACD":
                    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD", line=dict(color=CYAN)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=d["Signal"], name="Signal", line=dict(color=ACCENT)), row=2, col=1)
                    fig.add_trace(go.Bar(x=d.index, y=d["Histogram"], name="Histogram",
                                          marker_color=[GREEN if v >= 0 else RED for v in d["Histogram"].fillna(0)]),
                                  row=2, col=1)

                elif indicator_name == "CCI":
                    fig.add_trace(go.Scatter(x=d.index, y=d["CCI"], name="CCI", line=dict(color=PURPLE)), row=2, col=1)
                    fig.add_hline(y=100, line=dict(color=GREEN, dash="dash"), row=2, col=1)
                    fig.add_hline(y=-100, line=dict(color=RED, dash="dash"), row=2, col=1)
                    fig.add_hline(y=0, line=dict(color="#666", dash="dot"), row=2, col=1)

                elif indicator_name == "Estocástico (%K / %D)":
                    fig.add_trace(go.Scatter(x=d.index, y=d["%K"], name="%K", line=dict(color=CYAN)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=d["%D"], name="%D", line=dict(color=RED)), row=2, col=1)
                    fig.add_hline(y=80, line=dict(color=GREEN, dash="dash"), row=2, col=1)
                    fig.add_hline(y=20, line=dict(color=RED, dash="dash"), row=2, col=1)
                    fig.update_yaxes(range=[0, 100], row=2, col=1)

                elif indicator_name == "Money Flow Index":
                    fig.add_trace(go.Scatter(x=d.index, y=d["MFI"], name="MFI", line=dict(color=PURPLE)), row=2, col=1)
                    fig.add_hline(y=80, line=dict(color=RED, dash="dash"), row=2, col=1)
                    fig.add_hline(y=20, line=dict(color=GREEN, dash="dash"), row=2, col=1)
                    fig.update_yaxes(range=[0, 100], row=2, col=1)

                elif indicator_name == "On Balance Volume":
                    fig.add_trace(go.Scatter(x=d.index, y=d["OBV"], name="OBV", line=dict(color=PURPLE)), row=2, col=1)

                elif indicator_name == "Aroon":
                    fig.add_trace(go.Scatter(x=d.index, y=d["Aroon Up"], name="Up", line=dict(color=GREEN)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=d["Aroon Down"], name="Down", line=dict(color=RED)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=d["Aroon Oscillator"], name="Osc",
                                              line=dict(color=PURPLE, dash="dash")), row=2, col=1)
                    fig.add_hline(y=0, line=dict(color="#666"), row=2, col=1)

                elif indicator_name == "OAD (Williams)":
                    fig.add_trace(go.Scatter(x=d.index, y=d["OAD"], name="OAD", line=dict(color=PURPLE)), row=2, col=1)
                    fig.add_trace(go.Scatter(x=d.index, y=d["OAD Signal"], name="Signal", line=dict(color=ACCENT)), row=2, col=1)
                    fig.add_hline(y=70, line=dict(color=RED, dash="dash"), row=2, col=1)
                    fig.add_hline(y=30, line=dict(color=GREEN, dash="dash"), row=2, col=1)
                    fig.update_yaxes(range=[-5, 105], row=2, col=1)

                elif indicator_name == "Momentum / Impulso":
                    fig.add_trace(go.Scatter(x=d.index, y=d["Momentum"], name="Momentum", line=dict(color=ACCENT)), row=2, col=1)
                    fig.add_hline(y=0, line=dict(color="#666", dash="dash"), row=2, col=1)

                elif indicator_name == "Hurst (rolling)":
                    fig.add_trace(go.Scatter(x=d.index, y=d["Hurst"], name="Hurst", line=dict(color=PURPLE)), row=2, col=1)
                    fig.add_hline(y=0.5, line=dict(color=ACCENT, dash="dash"), row=2, col=1,
                                   annotation_text="H=0.5 (random walk)", annotation_position="top right")
                    fig.update_yaxes(range=[0, 1], row=2, col=1)

                fig.update_layout(
                    **PLOTLY_LAYOUT, height=700, xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            elif kind == "elliott":
                d, picos, valles = ind.elliott(df, **params)
                waves = ind.label_elliott_waves(picos, valles, d["Suavizado"].values)

                fig = go.Figure()
                # Precio real (sutil)
                fig.add_trace(go.Scatter(
                    x=d.index, y=d["close"], name="Close",
                    line=dict(color="#444", width=1), hovertemplate="%{y:.2f}<extra></extra>",
                ))
                # Curva suavizada (la línea sobre la que se identifican ondas)
                fig.add_trace(go.Scatter(
                    x=d.index, y=d["Suavizado"], name="Smoothed (Savitzky-Golay)",
                    line=dict(color=CYAN, width=2),
                    hovertemplate="%{y:.2f}<extra></extra>",
                ))

                # ── Líneas conectando pivotes en orden cronológico (path de la onda)
                if waves:
                    path_x = [d.index[w["idx"]] for w in waves]
                    path_y = [w["price"] for w in waves]
                    fig.add_trace(go.Scatter(
                        x=path_x, y=path_y, mode="lines",
                        line=dict(color="#888", width=1, dash="dot"),
                        name="Wave path", hoverinfo="skip", showlegend=False,
                    ))

                # ── Pivotes etiquetados por grupo: impulso 1/3/5, retroceso 2/4, corrección A/B/C
                IMPULSE_MAIN = {"1", "3", "5"}
                IMPULSE_CORR = {"2", "4"}
                ABC = {"A", "B", "C"}

                def _draw_group(labels_set: set, color: str, name: str, symbol: str = "circle"):
                    xs, ys, texts, invalid = [], [], [], []
                    for w in waves:
                        if w["label"] in labels_set:
                            xs.append(d.index[w["idx"]])
                            ys.append(w["price"])
                            tag = w["label"]
                            if not w["valid"]:
                                tag = tag + "?"  # marca rule violation
                            texts.append(f"<b>{tag}</b>")
                            invalid.append(not w["valid"])
                    if xs:
                        fig.add_trace(go.Scatter(
                            x=xs, y=ys, mode="markers+text",
                            marker=dict(
                                color=color, size=16, symbol=symbol,
                                line=dict(color="#000", width=1.5),
                            ),
                            text=texts, textposition="top center",
                            textfont=dict(
                                color=color, size=14, family="JetBrains Mono",
                            ),
                            name=name,
                            hovertemplate="<b>Wave %{text}</b><br>%{y:.2f}<extra></extra>",
                        ))

                _draw_group(IMPULSE_MAIN, GREEN, "Impulse 1·3·5", symbol="circle")
                _draw_group(IMPULSE_CORR, "#ff9500", "Pullback 2·4", symbol="diamond")
                _draw_group(ABC, RED, "Correction A·B·C", symbol="square")

                fig.update_layout(
                    **PLOTLY_LAYOUT, height=620,
                    xaxis_rangeslider_visible=False,
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                # ── Resumen de etiquetado y validaciones
                if waves:
                    n_waves = len(waves)
                    n_invalid = sum(1 for w in waves if not w["valid"])
                    last_label = waves[-1]["label"]
                    n_cycles = n_waves // 8
                    next_label = ["1", "2", "3", "4", "5", "A", "B", "C"][n_waves % 8]

                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("PIVOTS DETECTED", n_waves)
                    cc2.metric("COMPLETED 8-WAVE CYCLES", n_cycles)
                    cc3.metric("LAST WAVE", last_label)
                    cc4.metric("RULE VIOLATIONS", n_invalid,
                               help="Waves marked with '?' breach Elliott rules "
                                    "(wave 2 retracement, wave 3 not shortest, wave 4 overlap with wave 1).")

                    st.caption(
                        f"Next expected pivot: **{next_label}**  ·  "
                        "1·3·5 = impulse in trend direction (green) · "
                        "2·4 = corrective pullback within impulse (orange) · "
                        "A·B·C = corrective phase after impulse (red) · "
                        "Suffix '?' = rule violation."
                    )
                else:
                    st.caption("No pivots detected. Reduce 'prominencia' or 'distancia' to be more sensitive.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB · FUNDAMENTALES
# ──────────────────────────────────────────────────────────────────────────────
with tab_fund:
    st.caption("ℹ️ Datos de “foto”: ratios TTM (últimos 12 meses) y estados contables del "
               "último reporte. **No dependen del período del sidebar** — ese control no afecta este tab.")
    sub1, sub2 = st.tabs(["RATIOS TTM", "STATEMENTS"])

    with sub1:
        st.markdown("##### TTM RATIOS COMPARISON")
        with st.spinner("Fetching ratios from FMP..."):
            df_ratios = fetch_ratios(tickers)

        if df_ratios is None or df_ratios.empty:
            st.warning("FMP returned no ratios for the requested tickers.")
        else:
            st.dataframe(df_ratios.round(2), use_container_width=True)

            ratios_list = list(df_ratios.index)
            fig = make_subplots(rows=2, cols=2, subplot_titles=ratios_list)
            colors_cycle = [ACCENT, CYAN, GREEN, RED, PURPLE, "#ffa600", "#00ffaa"]
            for i, ratio_name in enumerate(ratios_list):
                r, c = i // 2 + 1, i % 2 + 1
                vals = pd.to_numeric(df_ratios.loc[ratio_name], errors="coerce")
                fig.add_trace(
                    go.Bar(
                        x=vals.index, y=vals.values, name=ratio_name,
                        marker=dict(color=colors_cycle[: len(vals)]),
                        text=[f"{v:.1f}" if pd.notna(v) else "—" for v in vals.values],
                        textposition="outside", showlegend=False,
                    ),
                    row=r, col=c,
                )
            fig.update_layout(**PLOTLY_LAYOUT, height=600)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with sub2:
        st.markdown("##### FINANCIAL STATEMENTS")
        col1, col2 = st.columns([1, 1])
        with col1:
            statement = st.selectbox(
                "Statement",
                list(fnd.STATEMENTS.keys()),
                index=1,
            )
        with col2:
            period = st.selectbox("Period", ["annual", "quarter"], index=0)

        with st.spinner(f"Fetching {statement} ({period})..."):
            df_st = fetch_statement(tickers, statement, period)

        if df_st is None or df_st.empty:
            st.warning("No data.")
        else:
            # Para visualización, mostramos columnas numéricas formateadas
            df_view = df_st.copy()
            num_cols = df_view.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                df_view[c] = df_view[c].apply(fnd.format_amount)
            st.dataframe(df_view, use_container_width=True, hide_index=True)

            # Botón para descargar Excel
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df_st.to_excel(writer, sheet_name=statement[:31], index=False)
            st.download_button(
                "EXPORT TO EXCEL",
                data=buf.getvalue(),
                file_name=f"{statement.replace(' ', '_')}_{period}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB · THEORY  (Clase 3 — Markowitz, CAPM, Frontera, CML)
# ──────────────────────────────────────────────────────────────────────────────
with tab_theory:
    # ── Armar el panel de precios alineado y limpio ──────────────────────
    # 1) Sólo tickers con datos no vacíos
    # 2) Alineamos por fecha intersección (.dropna()) para que cov/optim
    #    no reciban NaN — SLSQP no converge con NaN
    # adjclose (ajustado por dividendos/splits) → necesario para CAPM y
    # frontera eficiente. Con close crudo, betas y retornos esperados
    # quedan sesgados para tickers que pagan dividendos.
    closes_all = pd.DataFrame(
        {tk: _adj_price(df) for tk, df in history.items()
         if isinstance(df, pd.DataFrame) and not df.empty and "close" in df.columns}
    )
    # Eliminar tickers que quedaron todo NaN (sin overlap)
    closes_all = closes_all.dropna(axis=1, how="all")
    # Alinear todas las series a la intersección de fechas
    closes_all = closes_all.dropna(axis=0, how="any")

    if closes_all.shape[1] < 2:
        st.warning(
            "Need at least 2 tickers with overlapping price history. "
            "Check the watchlist and the selected date range."
        )
    elif closes_all.shape[0] < 30:
        st.warning(
            f"Only {closes_all.shape[0]} aligned observations — too few for "
            "reliable Σ/correlation/optimization. Widen the date range."
        )
    else:
        rets_log = pt.log_returns(closes_all).dropna()
        rets_simple = pt.simple_returns(closes_all).dropna()

        st.caption(
            f"Returns computed from {closes_all.shape[0]} aligned observations × "
            f"{closes_all.shape[1]} assets. Period: {closes_all.index[0].date()} "
            f"→ {closes_all.index[-1].date()}."
        )

        th1, th2, th3, th4, th5, th6 = st.tabs([
            "COVARIANCE", "CORRELATION", "STABILITY",
            "CAPM / SML", "FRONTIER", "UPSIDE",
        ])

        # ═══════════════════════════════════════════════════════════════════
        # SUB-TAB · COVARIANCE
        # ═══════════════════════════════════════════════════════════════════
        with th1:
            st.markdown("##### VARIANCE-COVARIANCE MATRIX Σ")
            cov_d = pt.cov_matrix(rets_log, annualize=False)
            cov_a = pt.cov_matrix(rets_log, annualize=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Σ DAILY (log-returns)**")
                fig_cd = go.Figure(go.Heatmap(
                    z=cov_d.values, x=cov_d.columns, y=cov_d.index,
                    colorscale="Blues", text=cov_d.round(6).values,
                    texttemplate="%{text}", textfont={"size": 9},
                    colorbar=dict(title="Cov<br>(daily)"),
                ))
                fig_cd.update_layout(**PLOTLY_LAYOUT, height=420)
                st.plotly_chart(fig_cd, use_container_width=True, config=PLOTLY_CONFIG)
            with cc2:
                st.markdown("**Σ ANNUALIZED (×252)**")
                fig_ca = go.Figure(go.Heatmap(
                    z=cov_a.values, x=cov_a.columns, y=cov_a.index,
                    colorscale="Oranges", text=cov_a.round(4).values,
                    texttemplate="%{text}", textfont={"size": 9},
                    colorbar=dict(title="Cov<br>(annual)"),
                ))
                fig_ca.update_layout(**PLOTLY_LAYOUT, height=420)
                st.plotly_chart(fig_ca, use_container_width=True, config=PLOTLY_CONFIG)

            # Diagonal: σ² y σ
            st.markdown("**ANNUALIZED VOLATILITY (diagonal of Σ)**")
            diag = pd.DataFrame({
                "Variance (σ²)": np.diag(cov_a),
                "Volatility (σ)": np.sqrt(np.diag(cov_a)),
            }, index=cov_a.index)
            diag["Volatility (σ)"] = diag["Volatility (σ)"].apply(lambda v: f"{v:.2%}")
            diag["Variance (σ²)"] = diag["Variance (σ²)"].apply(lambda v: f"{v:.4f}")
            st.dataframe(diag, use_container_width=True)

            st.caption(
                "Variance on the diagonal · Covariance off-diagonal · "
                "Annualized = daily × 252 (assumes daily independence)."
            )

        # ═══════════════════════════════════════════════════════════════════
        # SUB-TAB · CORRELATION
        # ═══════════════════════════════════════════════════════════════════
        with th2:
            st.markdown("##### CORRELATION MATRIX")
            cmethod = st.radio(
                "Method",
                ["pearson", "spearman", "kendall"],
                index=0, horizontal=True,
                help="Pearson = linear · Spearman = monotonic (ranks) · "
                     "Kendall = pairwise concordance.",
            )
            corr = pt.correlation_matrix(rets_log, method=cmethod)

            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="RdYlGn", zmin=-1, zmax=1, zmid=0,
                text=corr.round(3).values, texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="ρ"),
            ))
            fig_corr.update_layout(**PLOTLY_LAYOUT, height=500)
            st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

            # Pares máximos/mínimos
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            upper = corr.where(mask).stack()
            if not upper.empty:
                max_pair = upper.idxmax()
                min_pair = upper.idxmin()
                cc1, cc2 = st.columns(2)
                cc1.metric(
                    "MOST CORRELATED PAIR",
                    f"{max_pair[0]} ↔ {max_pair[1]}",
                    delta=f"ρ = {upper.loc[max_pair]:+.3f}",
                )
                cc2.metric(
                    "LEAST CORRELATED PAIR",
                    f"{min_pair[0]} ↔ {min_pair[1]}",
                    delta=f"ρ = {upper.loc[min_pair]:+.3f}",
                )

            st.markdown("---")
            st.markdown("##### LINEAR vs NON-LINEAR DEPENDENCE")
            st.caption(
                "Pearson only captures LINEAR relationships. Below, four canonical "
                "cases show where r ≈ 0 hides strong dependence."
            )
            # Generamos los 4 casos canónicos (idéntico a 03_Pearson_vs_NoLineal.py)
            rng = np.random.default_rng(42)
            N = 500
            x = np.linspace(-3, 3, N)
            noise = rng.normal(0, 0.05 * np.std(x), N)
            theta = np.linspace(0, 2 * np.pi, N)
            cases = [
                ("Linear positive · y = 2x", x, 2 * x + noise * 6),
                ("Linear negative · y = −2x", x, -2 * x + noise * 6),
                ("Quadratic · y = x²", x, x ** 2 + noise * 6),
                ("Circle · x²+y² = 1", np.cos(theta) + rng.normal(0, 0.04, N),
                                       np.sin(theta) + rng.normal(0, 0.04, N)),
            ]
            # Pre-calculamos correlaciones y las metemos en los títulos de subplot
            # (más limpio que add_annotation con conflictos de refs).
            corr_per_case = [pt.correlation_kinds(c[1], c[2]) for c in cases]
            titles = [
                (f"<b>{c[0]}</b><br>"
                 f"<span style='font-size:10px;color:#d97a00'>"
                 f"P={r['pearson']:+.2f} · S={r['spearman']:+.2f} · K={r['kendall']:+.2f}"
                 f"</span>")
                for c, r in zip(cases, corr_per_case)
            ]
            fig_nl = make_subplots(rows=2, cols=2, subplot_titles=titles,
                                    vertical_spacing=0.18, horizontal_spacing=0.10)
            for i, (name, xi, yi) in enumerate(cases):
                row, col = i // 2 + 1, i % 2 + 1
                fig_nl.add_trace(
                    go.Scatter(
                        x=xi, y=yi, mode="markers",
                        marker=dict(color=ACCENT, size=4, opacity=0.6),
                        showlegend=False,
                        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                    ),
                    row=row, col=col,
                )
            fig_nl.update_layout(**PLOTLY_LAYOUT, height=640, showlegend=False)
            # Estilo de los títulos (color ámbar, monoespacio)
            fig_nl.update_annotations(font=dict(family="JetBrains Mono", size=11,
                                                  color="#d97a00"))
            st.plotly_chart(fig_nl, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption(
                "r ≈ 0 does NOT imply independence — only absence of LINEAR relationship. "
                "Spearman / Kendall catch some non-linear cases but the circle defeats them all."
            )

        # ═══════════════════════════════════════════════════════════════════
        # SUB-TAB · STABILITY (Frobenius)
        # ═══════════════════════════════════════════════════════════════════
        with th3:
            st.markdown("##### TEMPORAL STABILITY · FROBENIUS DISTANCE")
            n_periods = st.slider("Number of sub-periods", 2, 8, 2, step=1)
            corrs_split = pt.split_correlations(rets_log, n_periods=n_periods)

            cols = st.columns(min(n_periods, 4))
            for i, c in enumerate(corrs_split):
                with cols[i % len(cols)]:
                    fig = go.Figure(go.Heatmap(
                        z=c.values, x=c.columns, y=c.index,
                        colorscale="RdYlGn", zmin=-1, zmax=1, zmid=0,
                        text=c.round(2).values, texttemplate="%{text}",
                        textfont={"size": 9}, showscale=(i == 0),
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                                       title=dict(text=f"Period {i+1}", font=dict(color=ACCENT, size=12)))
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

            # Distancias entre períodos consecutivos
            st.markdown("**FROBENIUS DISTANCES**")
            dist_rows = []
            for i in range(len(corrs_split) - 1):
                d = pt.frobenius_distance(corrs_split[i], corrs_split[i + 1])
                dist_rows.append({"From → To": f"P{i+1} → P{i+2}", "Distance": d})
            n = closes_all.shape[1]
            id_dist = pt.frobenius_distance(corrs_split[0], np.eye(n))
            dist_rows.append({"From → To": "P1 → Identity (independence)", "Distance": id_dist})
            dist_df = pd.DataFrame(dist_rows)
            dist_df["Distance"] = dist_df["Distance"].apply(lambda v: f"{v:.4f}")
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            st.caption(
                "‖A − B‖_F = √Σ(aᵢⱼ − bᵢⱼ)². Distance to the identity matrix gives "
                "a reference scale: closer to 0 means more stable correlation structure."
            )

        # ═══════════════════════════════════════════════════════════════════
        # SUB-TAB · CAPM / SML
        # ═══════════════════════════════════════════════════════════════════
        with th4:
            st.markdown("##### CAPM · SECURITY MARKET LINE")
            ccm1, ccm2 = st.columns([2, 1])
            with ccm1:
                bm_options = [t for t in closes_all.columns] + ["^GSPC", "^MERV", "SPY"]
                bm_options = list(dict.fromkeys(bm_options))  # de-dup
                benchmark = st.selectbox("Benchmark", bm_options,
                                          index=bm_options.index("^GSPC") if "^GSPC" in bm_options else 0)
            with ccm2:
                rf_capm = st.number_input("R_f (annual)", min_value=0.0, max_value=0.30,
                                            value=0.04, step=0.005, format="%.3f", key="capm_rf")

            # Si el benchmark no está en la watchlist, lo agregamos al fetch
            closes_with_bm = None
            if benchmark in closes_all.columns:
                closes_with_bm = closes_all
            else:
                try:
                    bm_hist = fetch_history(
                        (benchmark,),
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        interval=interval_choice,
                    )
                    bm_df = bm_hist.get(benchmark)
                    if bm_df is not None and not bm_df.empty and "close" in bm_df.columns:
                        closes_with_bm = closes_all.join(
                            _adj_price(bm_df).rename(benchmark),
                            how="inner",
                        ).dropna()
                    else:
                        st.error(f"Could not fetch benchmark '{benchmark}'.")
                except Exception as e:
                    st.error(f"Error fetching benchmark '{benchmark}': {e}")

            if closes_with_bm is not None and benchmark in closes_with_bm.columns \
               and closes_with_bm.shape[0] > 10:
                rets_bm = pt.simple_returns(closes_with_bm).dropna()
                try:
                    capm_df = pt.capm_analysis(rets_bm, benchmark=benchmark, rf=rf_capm)
                except (ValueError, ZeroDivisionError) as e:
                    st.error(f"CAPM error: {e}")
                    capm_df = None

                if capm_df is not None and not capm_df.empty:
                    # NaN guard: dropea filas con NaN antes de graficar
                    capm_df = capm_df.dropna()
                    # Tabla
                    df_show = capm_df.copy()
                    for c in ["annual_return", "annual_vol", "capm_return", "alpha"]:
                        df_show[c] = df_show[c].apply(lambda v: f"{v:+.2%}")
                    df_show["beta"] = df_show["beta"].apply(lambda v: f"{v:.3f}")
                    st.dataframe(df_show, use_container_width=True)

                    mkt_return = pt.annualize_return_from_daily(rets_bm[benchmark])
                    mkt_premium = mkt_return - rf_capm
                    st.caption(
                        f"Market ({benchmark}): E[Rm] = {mkt_return:+.2%}  ·  "
                        f"R_f = {rf_capm:.2%}  ·  Premium = {mkt_premium:+.2%}"
                    )

                    # SML
                    betas_range = np.linspace(0, max(capm_df["beta"].max(), 1.5) * 1.1, 100)
                    sml = pt.sml_line(rf_capm, mkt_return, betas_range)

                    fig_sml = go.Figure()
                    fig_sml.add_trace(go.Scatter(
                        x=betas_range, y=sml, mode="lines",
                        line=dict(color=ACCENT, width=2.5), name="SML",
                        hovertemplate="β=%{x:.2f}<br>E[R]=%{y:.2%}<extra></extra>",
                    ))
                    fig_sml.add_trace(go.Scatter(
                        x=[0], y=[rf_capm], mode="markers",
                        marker=dict(color=ACCENT, size=12, symbol="diamond"),
                        name=f"R_f = {rf_capm:.1%}",
                        hovertemplate=f"R_f = {rf_capm:.2%}<extra></extra>",
                    ))
                    fig_sml.add_trace(go.Scatter(
                        x=[1], y=[mkt_return], mode="markers",
                        marker=dict(color=CYAN, size=14, symbol="square"),
                        name=f"Market ({benchmark})",
                        hovertemplate=f"Market<br>β=1<br>E[R]={mkt_return:+.2%}<extra></extra>",
                    ))
                    for tk in capm_df.index:
                        a = capm_df.loc[tk, "alpha"]
                        col = GREEN if a > 0 else RED
                        fig_sml.add_trace(go.Scatter(
                            x=[capm_df.loc[tk, "beta"]],
                            y=[capm_df.loc[tk, "annual_return"]],
                            mode="markers+text",
                            marker=dict(color=col, size=14,
                                         line=dict(color="#000", width=1)),
                            text=[tk], textposition="top center",
                            textfont=dict(color=col, family="JetBrains Mono", size=11),
                            name=tk, showlegend=False,
                            hovertemplate=(f"<b>{tk}</b><br>β=%{{x:.3f}}<br>"
                                           f"E[R]=%{{y:+.2%}}<br>α={a:+.2%}<extra></extra>"),
                        ))
                    fig_sml.update_layout(
                        **PLOTLY_LAYOUT, height=500,
                        xaxis_title="Beta (β)", yaxis_title="Annual return",
                    )
                    fig_sml.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig_sml, use_container_width=True, config=PLOTLY_CONFIG)
                    st.caption(
                        "Green = positive α (above the SML, outperformed CAPM) · "
                        "Red = negative α · Vertical dotted move = α magnitude."
                    )

        # ═══════════════════════════════════════════════════════════════════
        # SUB-TAB · FRONTIER  (Monte Carlo + SLSQP + CML)
        # ═══════════════════════════════════════════════════════════════════
        with th5:
            st.markdown("##### EFFICIENT FRONTIER · MONTE CARLO + SLSQP + CML")

            cf1, cf2, cf3, cf4 = st.columns(4)
            n_mc = cf1.number_input("MC simulations", min_value=1000, max_value=100_000,
                                     value=10_000, step=1000)
            rf_fr = cf2.number_input("R_f (annual)", min_value=0.0, max_value=0.30,
                                      value=0.04, step=0.005, format="%.3f", key="frontier_rf")
            bound_max = cf3.number_input("Max weight per asset", min_value=0.05, max_value=1.0,
                                          value=1.0, step=0.05, format="%.2f")
            allow_short = cf4.checkbox("Allow shorts", value=False)
            bound_min = -1.0 if allow_short else 0.0

            rets_simple_th = pt.simple_returns(closes_all).dropna()
            ann_ret = pt.annualize_return_from_daily(rets_simple_th).dropna()
            # Reindexamos cov al mismo set de tickers que ann_ret (sin NaN)
            valid = ann_ret.index.tolist()
            ann_cov = pt.cov_matrix(rets_simple_th[valid], annualize=True)

            mc = ef = tan = None
            if len(valid) < 2:
                st.error("Not enough valid tickers for optimization.")
            else:
                try:
                    with st.spinner("Running Monte Carlo + SLSQP optimization..."):
                        mc = pt.monte_carlo_frontier(ann_ret, ann_cov,
                                                      n_portfolios=int(n_mc), rf=rf_fr)
                        ef = pt.efficient_frontier(ann_ret, ann_cov, n_points=100,
                                                    bound_min=bound_min, bound_max=bound_max)
                        tan = pt.tangent_portfolio(ann_ret, ann_cov, rf=rf_fr,
                                                    bound_min=bound_min, bound_max=bound_max)
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

            # Sólo seguimos si el optimizador terminó OK
            if tan is not None and mc is not None and ef is not None:
                if not tan.get("success"):
                    st.warning(
                        "SLSQP did not converge cleanly. Result is approximate — "
                        "try a wider date range or different constraints."
                    )

                # KPIs del tangente
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("TANGENT RETURN", f"{tan['return']:+.2%}")
                k2.metric("TANGENT VOL", f"{tan['vol']:.2%}")
                k3.metric("SHARPE", f"{tan['sharpe']:.3f}")
                k4.metric("CONSTRAINT", f"[{bound_min:.0%}, {bound_max:.0%}]")

                # Chart principal: nube MC + frontera SLSQP + CML + tangente
                fig_fr = go.Figure()
                fig_fr.add_trace(go.Scatter(
                    x=mc["results"][1], y=mc["results"][0], mode="markers",
                    marker=dict(
                        color=mc["results"][2], colorscale="Viridis", size=4,
                        opacity=0.5, colorbar=dict(title="Sharpe"),
                    ),
                    name="Monte Carlo",
                    hovertemplate="σ=%{x:.2%}<br>E[R]=%{y:+.2%}<br>Sharpe=%{marker.color:.2f}<extra></extra>",
                ))
                # Frontera SLSQP
                ef_ok = ef.dropna(subset=["vol"])
                fig_fr.add_trace(go.Scatter(
                    x=ef_ok["vol"], y=ef_ok["target_return"], mode="lines",
                    line=dict(color=ACCENT, width=2.5, dash="dash"),
                    name="Efficient frontier (SLSQP)",
                    hovertemplate="σ=%{x:.2%}<br>E[R]=%{y:+.2%}<extra></extra>",
                ))
                # CML
                sigma_max = max(float(mc["results"][1].max()),
                                 float(ef_ok["vol"].max()) if not ef_ok.empty else tan["vol"])
                sigmas, cml_y = pt.cml_points(rf_fr, tan["sharpe"], sigma_max * 1.1, n=100)
                fig_fr.add_trace(go.Scatter(
                    x=sigmas, y=cml_y, mode="lines",
                    line=dict(color=CYAN, width=2),
                    name="Capital Market Line",
                    hovertemplate="σ=%{x:.2%}<br>E[R]=%{y:+.2%}<extra></extra>",
                ))
                # R_f
                fig_fr.add_trace(go.Scatter(
                    x=[0], y=[rf_fr], mode="markers",
                    marker=dict(color=ACCENT, size=14, symbol="diamond"),
                    name=f"R_f = {rf_fr:.1%}",
                    hovertemplate=f"R_f = {rf_fr:.2%}<extra></extra>",
                ))
                # Tangente
                fig_fr.add_trace(go.Scatter(
                    x=[tan["vol"]], y=[tan["return"]], mode="markers",
                    marker=dict(color=GREEN, size=22, symbol="star",
                                 line=dict(color="#000", width=1.5)),
                    name=f"Tangent (Sharpe={tan['sharpe']:.2f})",
                    hovertemplate=(f"<b>Tangent</b><br>σ={tan['vol']:.2%}<br>"
                                   f"E[R]={tan['return']:+.2%}<br>"
                                   f"Sharpe={tan['sharpe']:.3f}<extra></extra>"),
                ))
                # Activos individuales
                stock_vol = pt.annualize_vol_from_daily(rets_simple_th)
                for tk in ann_ret.index:
                    fig_fr.add_trace(go.Scatter(
                        x=[stock_vol[tk]], y=[ann_ret[tk]],
                        mode="markers+text",
                        marker=dict(color=PURPLE, size=10,
                                     line=dict(color="#000", width=1)),
                        text=[tk], textposition="top center",
                        textfont=dict(color=PURPLE, family="JetBrains Mono", size=10),
                        showlegend=False,
                        hovertemplate=(f"<b>{tk}</b><br>σ=%{{x:.2%}}<br>"
                                       f"E[R]=%{{y:+.2%}}<extra></extra>"),
                    ))
                fig_fr.update_layout(
                    **PLOTLY_LAYOUT, height=600,
                    xaxis_title="Volatility σ", yaxis_title="Expected return E[R]",
                )
                fig_fr.update_xaxes(tickformat=".0%")
                fig_fr.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_fr, use_container_width=True, config=PLOTLY_CONFIG)

                # Composición y contribución al riesgo del tangente
                cw1, cw2 = st.columns(2)
                with cw1:
                    st.markdown("**TANGENT WEIGHTS**")
                    w_df = pd.DataFrame({
                        "Ticker": tan["tickers"],
                        "Weight": tan["weights"],
                    })
                    fig_w = go.Figure(go.Bar(
                        x=w_df["Ticker"], y=w_df["Weight"] * 100,
                        marker_color=ACCENT,
                        text=[f"{w:.1%}" for w in w_df["Weight"]],
                        textposition="outside",
                    ))
                    fig_w.update_layout(**PLOTLY_LAYOUT, height=350,
                                         yaxis_title="Weight %", showlegend=False)
                    st.plotly_chart(fig_w, use_container_width=True, config=PLOTLY_CONFIG)
                with cw2:
                    st.markdown("**RISK CONTRIBUTION**")
                    rc = pt.risk_contribution(tan["weights"], ann_cov)
                    fig_rc = go.Figure(go.Bar(
                        x=rc.index, y=rc.values,
                        marker_color=RED,
                        text=[f"{v:.1f}%" for v in rc.values],
                        textposition="outside",
                    ))
                    fig_rc.update_layout(**PLOTLY_LAYOUT, height=350,
                                          yaxis_title="Risk contribution %", showlegend=False)
                    st.plotly_chart(fig_rc, use_container_width=True, config=PLOTLY_CONFIG)

                st.caption(
                    "Monte Carlo cloud approximates the feasible set. SLSQP gives the TRUE "
                    "frontier (envelope). CML is tangent to the frontier at the tangent "
                    "portfolio — its slope = max Sharpe."
                )

        # ═══════════════════════════════════════════════════════════════════
        # SUB-TAB · UPSIDE FRONTIER (vistas propias)
        # ═══════════════════════════════════════════════════════════════════
        with th6:
            st.markdown("##### UPSIDE FRONTIER · CUSTOM EXPECTED RETURNS")
            st.caption(
                "Replace historical means with your own E[R] estimates (analyst views, "
                "forecast models). Σ comes from history. Garbage in → garbage out: the "
                "output is very sensitive to the inputs you provide here."
            )

            rets_simple_th = pt.simple_returns(closes_all).dropna()
            hist_er = pt.annualize_return_from_daily(rets_simple_th).fillna(0.0)
            ann_cov_v = pt.cov_matrix(rets_simple_th, annualize=True)

            st.markdown("**EXPECTED RETURNS E[R]** — edit the % per asset")
            tickers_in = list(closes_all.columns)
            cols_er = st.columns(min(len(tickers_in), 4))
            custom_er = {}
            for i, tk in enumerate(tickers_in):
                with cols_er[i % len(cols_er)]:
                    raw_default = hist_er.get(tk, 0.0)
                    # NaN-safe default
                    default = float(raw_default) * 100 if pd.notna(raw_default) else 0.0
                    default = max(-100.0, min(300.0, default))   # clamp al rango del input
                    custom_er[tk] = st.number_input(
                        tk, min_value=-100.0, max_value=300.0,
                        value=round(default, 2), step=1.0, format="%.2f",
                        key=f"er_{tk}",
                    ) / 100.0

            custom_er_s = pd.Series(custom_er).reindex(tickers_in)

            cu1, cu2, cu3 = st.columns(3)
            rf_up = cu1.number_input("R_f (annual)", min_value=0.0, max_value=0.30,
                                      value=0.04, step=0.005, format="%.3f", key="upside_rf")
            bound_max_up = cu2.number_input("Max weight per asset",
                                              min_value=0.05, max_value=1.0,
                                              value=1.0, step=0.05, format="%.2f",
                                              key="upside_bmax",
                                              help="Set to 0.20 to replicate script 10 "
                                                   "(20% cap forces diversification).")
            allow_short_up = cu3.checkbox("Allow shorts", value=False, key="upside_short")
            bound_min_up = -1.0 if allow_short_up else 0.0

            ef_up = tan_up = None
            try:
                with st.spinner("Optimizing with custom views..."):
                    ef_up = pt.efficient_frontier(
                        custom_er_s, ann_cov_v, n_points=80,
                        bound_min=bound_min_up, bound_max=bound_max_up,
                    )
                    tan_up = pt.tangent_portfolio(
                        custom_er_s, ann_cov_v, rf=rf_up,
                        bound_min=bound_min_up, bound_max=bound_max_up,
                    )
            except Exception as e:
                st.error(f"Optimization failed: {e}")

            if tan_up is not None and ef_up is not None:
                # KPIs
                ku1, ku2, ku3, ku4 = st.columns(4)
                ku1.metric("TANGENT RETURN", f"{tan_up['return']:+.2%}")
                ku2.metric("TANGENT VOL", f"{tan_up['vol']:.2%}")
                ku3.metric("SHARPE", f"{tan_up['sharpe']:.3f}")
                top1 = max(tan_up["weights"])
                ku4.metric("MAX CONCENTRATION", f"{top1:.0%}",
                            help="Largest single-asset weight. >50% indicates extreme "
                                 "concentration — consider lowering the max weight bound.")

                # Frontera + CML + tangente
                fig_up = go.Figure()
                ef_ok_up = ef_up.dropna(subset=["vol"])
                fig_up.add_trace(go.Scatter(
                    x=ef_ok_up["vol"], y=ef_ok_up["target_return"], mode="lines",
                    line=dict(color=ACCENT, width=2.5, dash="dash"),
                    name="Efficient frontier",
                    hovertemplate="σ=%{x:.2%}<br>E[R]=%{y:+.2%}<extra></extra>",
                ))
                sigma_max_up = (float(ef_ok_up["vol"].max()) * 1.15
                                 if not ef_ok_up.empty
                                 else tan_up["vol"] * 1.5)
                sigmas_u, cml_y_u = pt.cml_points(rf_up, tan_up["sharpe"], sigma_max_up, n=100)
                fig_up.add_trace(go.Scatter(
                    x=sigmas_u, y=cml_y_u, mode="lines",
                    line=dict(color=CYAN, width=2),
                    name="CML",
                ))
                fig_up.add_trace(go.Scatter(
                    x=[0], y=[rf_up], mode="markers",
                    marker=dict(color=ACCENT, size=14, symbol="diamond"),
                    name=f"R_f = {rf_up:.1%}",
                ))
                fig_up.add_trace(go.Scatter(
                    x=[tan_up["vol"]], y=[tan_up["return"]], mode="markers",
                    marker=dict(color=GREEN, size=22, symbol="star",
                                 line=dict(color="#000", width=1.5)),
                    name=f"Tangent (Sharpe={tan_up['sharpe']:.2f})",
                ))
                stock_vol_up = pt.annualize_vol_from_daily(rets_simple_th)
                for tk in custom_er_s.index:
                    fig_up.add_trace(go.Scatter(
                        x=[stock_vol_up.get(tk, 0)], y=[custom_er_s[tk]],
                        mode="markers+text",
                        marker=dict(color=PURPLE, size=10,
                                     line=dict(color="#000", width=1)),
                        text=[tk], textposition="top center",
                        textfont=dict(color=PURPLE, family="JetBrains Mono", size=10),
                        showlegend=False,
                    ))
                fig_up.update_layout(
                    **PLOTLY_LAYOUT, height=550,
                    xaxis_title="Volatility σ", yaxis_title="Expected return E[R]",
                )
                fig_up.update_xaxes(tickformat=".0%")
                fig_up.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_up, use_container_width=True, config=PLOTLY_CONFIG)

                # Composición + contrib al riesgo
                cu_w1, cu_w2 = st.columns(2)
                with cu_w1:
                    st.markdown("**TANGENT WEIGHTS**")
                    fig_wu = go.Figure(go.Bar(
                        x=tan_up["tickers"], y=np.array(tan_up["weights"]) * 100,
                        marker_color=[RED if w > 0.30 else ACCENT for w in tan_up["weights"]],
                        text=[f"{w:.1%}" for w in tan_up["weights"]],
                        textposition="outside",
                    ))
                    fig_wu.update_layout(**PLOTLY_LAYOUT, height=350,
                                          yaxis_title="Weight %", showlegend=False)
                    st.plotly_chart(fig_wu, use_container_width=True, config=PLOTLY_CONFIG)
                with cu_w2:
                    st.markdown("**RISK CONTRIBUTION**")
                    rc_u = pt.risk_contribution(tan_up["weights"], ann_cov_v)
                    fig_rcu = go.Figure(go.Bar(
                        x=rc_u.index, y=rc_u.values,
                        marker_color=RED,
                        text=[f"{v:.1f}%" for v in rc_u.values],
                        textposition="outside",
                    ))
                    fig_rcu.update_layout(**PLOTLY_LAYOUT, height=350,
                                           yaxis_title="Risk contribution %", showlegend=False)
                    st.plotly_chart(fig_rcu, use_container_width=True, config=PLOTLY_CONFIG)


# ──────────────────────────────────────────────────────────────────────────────
# TAB · PORTFOLIO
# ──────────────────────────────────────────────────────────────────────────────
with tab_port:
    st.markdown("##### PORTFOLIO BUILDER")

    st.markdown("**WEIGHTS (%)**")
    cols = st.columns(min(len(tickers), 6))
    weights = {}
    default_w = round(100 / max(len(tickers), 1), 2)
    for i, tk in enumerate(tickers):
        with cols[i % len(cols)]:
            weights[tk] = st.number_input(tk, min_value=0.0, max_value=100.0,
                                           value=default_w, step=1.0, key=f"w_{tk}")

    total_w = sum(weights.values())
    if abs(total_w - 100) > 0.01:
        st.warning(f"Weights sum to {total_w:.2f}% (should be 100%). Auto-normalized.")
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    # Total returns → usamos adjclose (incluye dividendos/splits)
    closes = pd.DataFrame({tk: _adj_price(df) for tk, df in history.items() if not df.empty})
    if closes.empty:
        st.error("No prices to build the portfolio.")
    else:
        rets = closes.pct_change().dropna(how="all")
        port_ret = sum(rets[tk] * weights.get(tk, 0) for tk in rets.columns)
        port_cum = (1 + port_ret).cumprod()
        bench_cum = (1 + rets.mean(axis=1)).cumprod()

        c1, c2, c3, c4 = st.columns(4)
        ret_total = port_cum.iloc[-1] - 1 if not port_cum.empty else 0
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = (port_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
        dd = (port_cum / port_cum.cummax() - 1).min()
        c1.metric("TOTAL RETURN", f"{ret_total * 100:+.2f}%")
        c2.metric("ANN. VOL", f"{ann_vol * 100:.2f}%")
        c3.metric("SHARPE (rf=0)", f"{sharpe:.2f}")
        c4.metric("MAX DD", f"{dd * 100:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, name="Portfolio",
                                  line=dict(color=ACCENT, width=2.5)))
        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name="Equal-weight benchmark",
                                  line=dict(color=CYAN, width=1.5, dash="dash")))
        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                          yaxis_title="Value (base 1.0)")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Allocation pie + correlation heatmap
        cc1, cc2 = st.columns(2)
        with cc1:
            fig_pie = go.Figure(go.Pie(
                labels=list(weights.keys()), values=list(weights.values()),
                hole=0.55, marker=dict(colors=[ACCENT, CYAN, GREEN, RED, PURPLE, "#ff8800", "#00ffaa"][: len(weights)]),
            ))
            fig_pie.update_layout(**PLOTLY_LAYOUT, height=380)
            st.markdown("**ALLOCATION**")
            st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)
        with cc2:
            corr = rets.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="RdYlGn", zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
            ))
            fig_corr.update_layout(**PLOTLY_LAYOUT, height=380)
            st.markdown("**CORRELATION**")
            st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("**DRAWDOWN (%)**")
        dd_series = port_cum / port_cum.cummax() - 1
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values * 100,
                                     fill="tozeroy", line=dict(color=RED), name="Drawdown"))
        fig_dd.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title="DD %")
        st.plotly_chart(fig_dd, use_container_width=True, config=PLOTLY_CONFIG)


# ──────────────────────────────────────────────────────────────────────────────
# TAB · RISK  ·  Clase 4 — VaR/CVaR (3 métodos) + Performance Attribution
# ──────────────────────────────────────────────────────────────────────────────
with tab_risk:
    sub_var, sub_attr = st.tabs(["VaR / CVaR", "PERFORMANCE ATTRIBUTION"])

    # ═══ VaR / CVaR ═══════════════════════════════════════════════════════
    with sub_var:
        st.markdown("##### VALUE AT RISK · PARAMÉTRICO · HISTÓRICO · MONTE CARLO")
        c1, c2, c3 = st.columns(3)
        with c1:
            var_tickers = st.multiselect(
                "Activos (cartera)", options=list(tickers),
                default=list(tickers)[: min(5, len(tickers))], key="var_tk")
        with c2:
            conf = st.selectbox("Confianza", [0.90, 0.95, 0.99], index=1, key="var_conf")
        with c3:
            capital = st.number_input("Capital (USD)", min_value=1000.0,
                                      value=1_000_000.0, step=10000.0, key="var_cap")
        dias = st.slider("Horizonte (días de tenencia)", 1, 20, 1, key="var_dias")
        st.caption(f"Historia del **{start_date.date()}** al **{end_date.date()}** "
                   f"(período global del sidebar).")

        if len(var_tickers) < 2:
            st.warning("Elegí al menos 2 activos para construir la cartera.")
        else:
            with st.spinner("Descargando historia y calculando VaR..."):
                adjm = fetch_adjclose_matrix(tuple(var_tickers),
                                             start_date.strftime("%Y-%m-%d"),
                                             end_date.strftime("%Y-%m-%d"))
            if adjm.empty or adjm.shape[1] < 2:
                st.error("No se pudo descargar historia suficiente para estos activos.")
            else:
                rets = adjm.pct_change().dropna()
                w = np.array([1 / adjm.shape[1]] * adjm.shape[1])
                p = ra.var_parametrico(rets, w, conf, capital, dias)
                h = ra.var_historico(rets, w, conf, capital)
                m = ra.var_montecarlo(rets, w, conf, capital, dias)

                st.caption(f"Cartera equiponderada de {adjm.shape[1]} activos · "
                           f"{len(rets)} días · {conf:.0%} confianza · horizonte {dias}d")
                k1, k2, k3 = st.columns(3)
                for col, r, color in [(k1, p, ACCENT), (k2, h, CYAN), (k3, m, PURPLE)]:
                    col.markdown(
                        f"<div class='bbg-panel'><div class='bbg-panel-header' "
                        f"style='border-left:3px solid {color}'>{r['metodo'].upper()}</div>"
                        f"<div class='bbg-panel-body'>"
                        f"<div class='label'>VaR {conf:.0%}</div>"
                        f"<div class='value' style='font-size:1.3rem'>USD {r['var_usd']:,.0f}</div>"
                        f"<div class='down'>{r['var_pct']:.2%} del capital</div>"
                        f"<div class='label' style='margin-top:6px'>CVaR (Expected Shortfall)</div>"
                        f"<div class='value' style='font-size:1.1rem'>USD {r['cvar_usd']:,.0f}</div>"
                        f"<div class='down'>{r['cvar_pct']:.2%}</div>"
                        f"</div></div>", unsafe_allow_html=True)

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**COMPARACIÓN DE MÉTODOS**")
                    comp = ra.var_comparacion(rets, w, conf, capital)
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Bar(x=comp["Método"], y=comp["VaR (USD)"], name="VaR",
                                           marker_color=RED,
                                           text=[f"{v:,.0f}" for v in comp["VaR (USD)"]],
                                           textposition="outside"))
                    fig_c.add_trace(go.Bar(x=comp["Método"], y=comp["CVaR (USD)"], name="CVaR",
                                           marker_color="#7a0010",
                                           text=[f"{v:,.0f}" for v in comp["CVaR (USD)"]],
                                           textposition="outside"))
                    fig_c.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
                                        yaxis_title="Pérdida (USD)")
                    st.plotly_chart(fig_c, use_container_width=True, config=PLOTLY_CONFIG)
                with cc2:
                    st.markdown("**DISTRIBUCIÓN DE RETORNOS Y COLA DE RIESGO**")
                    ret_hist = h["ret_cartera"]
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Histogram(x=ret_hist, nbinsx=40, histnorm="probability density",
                                                 marker_color="#8B6F47",
                                                 opacity=0.55, name="Histórico (real)"))
                    xs = np.linspace(float(ret_hist.min()), float(ret_hist.max()), 300)
                    _sig = p["sigma_T"] if p["sigma_T"] > 0 else 1e-9
                    ys = np.exp(-0.5 * ((xs - p["mu_T"]) / _sig) ** 2) / (_sig * np.sqrt(2 * np.pi))
                    fig_d.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                               line=dict(color=ACCENT, width=2.2),
                                               name="Paramétrico (normal)"))
                    for r, color, dash in [(p, ACCENT, "dash"), (h, CYAN, "dot"), (m, PURPLE, "dashdot")]:
                        fig_d.add_vline(x=r["ret_var"], line=dict(color=color, width=1.6, dash=dash))
                    fig_d.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="Densidad",
                                        xaxis_title="Retorno")
                    st.plotly_chart(fig_d, use_container_width=True, config=PLOTLY_CONFIG)

                disc = comp["VaR (USD)"].max() - comp["VaR (USD)"].min()
                mas = comp.loc[comp["VaR (USD)"].idxmax(), "Método"]
                men = comp.loc[comp["VaR (USD)"].idxmin(), "Método"]
                st.caption(f"Más conservador: **{mas}** · Menos conservador: **{men}** · "
                           f"Discrepancia máxima: USD {disc:,.0f} "
                           f"({disc / comp['VaR (USD)'].min() * 100:.1f}% del menor). "
                           f"Peor día histórico observado: {h['peor_dia']:+.2%}.")

    # ═══ PERFORMANCE ATTRIBUTION ══════════════════════════════════════════
    with sub_attr:
        st.markdown("##### BRINSON — ATRIBUCIÓN DE EXCESO DE RETORNO (BHB vs BF)")
        st.caption("Editá los pesos y retornos por sector de cartera y benchmark. "
                   "BHB descompone en Asignación + Selección + Interacción; "
                   "BF en Asignación + Selección (absorbe la interacción). "
                   "Ambas suman exactamente el mismo exceso.")
        default_attr = pd.DataFrame({
            "Sector": ["Tech", "Healthcare", "Finance"],
            "Peso Cartera": [0.60, 0.20, 0.20],
            "Retorno Cartera": [0.050, 0.030, 0.045],
            "Peso Benchmark": [0.55, 0.20, 0.25],
            "Retorno Benchmark": [0.045, 0.030, 0.020],
        })
        edited = st.data_editor(default_attr, num_rows="dynamic", use_container_width=True,
                                key="attr_editor")
        edited = edited.dropna()
        sw_p, sw_b = edited["Peso Cartera"].sum(), edited["Peso Benchmark"].sum()
        if abs(sw_p - 1) > 0.01 or abs(sw_b - 1) > 0.01:
            st.warning(f"Los pesos deberían sumar 1 (cartera={sw_p:.2f}, benchmark={sw_b:.2f}).")

        if len(edited) >= 1 and sw_p > 0 and sw_b > 0:
            sec = edited["Sector"].astype(str).tolist()
            wp = dict(zip(sec, edited["Peso Cartera"]))
            rp = dict(zip(sec, edited["Retorno Cartera"]))
            wb = dict(zip(sec, edited["Peso Benchmark"]))
            rb = dict(zip(sec, edited["Retorno Benchmark"]))
            df_bhb, tb = ra.bhb_attribution(sec, wp, rp, wb, rb)
            df_bf, tf = ra.bf_attribution(sec, wp, rp, wb, rb)

            k1, k2, k3 = st.columns(3)
            k1.metric("RETORNO CARTERA", f"{tb['R_P']:+.2%}")
            k2.metric("RETORNO BENCHMARK", f"{tb['R_B']:+.2%}")
            k3.metric("EXCESO", f"{tb['exceso']:+.2%}")

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**BHB — DESCOMPOSICIÓN POR SECTOR**")
                fig_b = go.Figure()
                fig_b.add_trace(go.Bar(x=df_bhb["Sector"], y=df_bhb["Asignacion"] * 100,
                                       name="Asignación", marker_color=ACCENT))
                fig_b.add_trace(go.Bar(x=df_bhb["Sector"], y=df_bhb["Seleccion"] * 100,
                                       name="Selección", marker_color=CYAN))
                fig_b.add_trace(go.Bar(x=df_bhb["Sector"], y=df_bhb["Interaccion"] * 100,
                                       name="Interacción", marker_color="#8B6F47"))
                fig_b.update_layout(**PLOTLY_LAYOUT, height=380, barmode="relative",
                                    yaxis_title="Contribución (%)")
                st.plotly_chart(fig_b, use_container_width=True, config=PLOTLY_CONFIG)
            with cc2:
                st.markdown("**BHB vs BF — TOTALES AGREGADOS**")
                fig_t = go.Figure()
                cats = ["Asignación", "Selección", "Interacción", "TOTAL = Exceso"]
                fig_t.add_trace(go.Bar(x=cats, name="BHB", marker_color=ACCENT,
                                       y=[tb["asignacion"] * 100, tb["seleccion"] * 100,
                                          tb["interaccion"] * 100, tb["exceso"] * 100]))
                fig_t.add_trace(go.Bar(x=cats, name="BF", marker_color=RED,
                                       y=[tf["allocation"] * 100, tf["selection"] * 100,
                                          0, tf["exceso"] * 100]))
                fig_t.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
                                    yaxis_title="Contribución (%)")
                st.plotly_chart(fig_t, use_container_width=True, config=PLOTLY_CONFIG)

            show = df_bhb[["Sector", "Asignacion", "Seleccion", "Interaccion", "Total"]].copy()
            show = show.merge(df_bf[["Sector", "Allocation", "Selection"]], on="Sector")
            show.columns = ["Sector", "BHB Asig", "BHB Sel", "BHB Inter", "BHB Total",
                            "BF Asig", "BF Sel"]
            for c in show.columns[1:]:
                show[c] = show[c].apply(lambda v: f"{v:+.4%}")
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption("La asignación BF puede tener signo OPUESTO a la BHB para un mismo "
                       "sector: BF premia sobreponderar sectores que superan al benchmark "
                       "TOTAL, no solo los de retorno positivo.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB · STRATEGY LAB  ·  Clase 5 — Backtesting & Machine Learning
# ──────────────────────────────────────────────────────────────────────────────
with tab_lab:
    st.markdown("##### BACKTESTING LAB · TREND · MEAN REVERSION · MOMENTUM · REBALANCEO · ML")
    cfg1, cfg2 = st.columns([3, 1])
    with cfg1:
        lab_tickers = st.multiselect(
            "Universo", options=sorted(set(list(tickers) + uni.US_LARGECAPS[:30])),
            default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"], key="lab_tk")
    with cfg2:
        cost_bps = st.number_input("Costo (bps/trade)", min_value=0.0, value=15.0,
                                   step=5.0, key="lab_cost")
    cost = cost_bps / 10000.0
    # ── Ventana de backtest propia (el backtesting tiene necesidades temporales
    #    distintas: historia larga + múltiples regímenes + control del corte
    #    in-sample/out-of-sample). Por defecto usa el período global del sidebar.
    with st.expander("⏱ Ventana de backtest e in-sample/out-of-sample (avanzado)", expanded=False):
        usar_propio = st.checkbox(
            "Usar una ventana propia para el backtest (recomendado: historia larga y multi-régimen)",
            value=False, key="lab_own_window")
        if usar_propio:
            wca, wcb = st.columns(2)
            lab_start_eff = datetime.combine(
                wca.date_input("Desde (backtest)", value=datetime(2010, 1, 1).date(),
                               min_value=datetime(2001, 1, 1).date(), key="lab_bt_start"),
                datetime.min.time())
            lab_end_eff = datetime.combine(
                wcb.date_input("Hasta (backtest)", value=end_date.date(),
                               min_value=datetime(2001, 1, 1).date(), key="lab_bt_end"),
                datetime.min.time())
        else:
            lab_start_eff, lab_end_eff = start_date, end_date
        is_pct = st.slider(
            "In-sample (% para entrenar/optimizar; el resto es out-of-sample)",
            50, 90, 80, step=5, key="lab_isfrac",
            help="Punto de corte IS/OOS para Trend/RSI/Momentum/Rebalanceo. "
                 "Las estrategias de Machine Learning usan walk-forward y no este corte.")
        is_frac = is_pct / 100.0

    _src = "ventana propia" if usar_propio else "período global del sidebar"
    st.caption(f"Backtest del **{lab_start_eff.date()}** al **{lab_end_eff.date()}** ({_src}) · "
               f"corte in-sample **{is_pct}%** / out-of-sample **{100-is_pct}%**. "
               f"Para resultados robustos conviene historia larga (5A+/MÁX).")

    strat = st.radio("Estrategia",
                     ["Trend (SMA/LMA)", "Mean Reversion (RSI)", "Momentum",
                      "Rebalanceo", "Machine Learning"],
                     horizontal=True, key="lab_strat")

    if len(lab_tickers) < 1:
        st.warning("Elegí al menos un activo.")
    else:
        with st.spinner("Descargando precios..."):
            adj = fetch_adjclose_matrix(tuple(lab_tickers),
                                        lab_start_eff.strftime("%Y-%m-%d"),
                                        lab_end_eff.strftime("%Y-%m-%d"))
        if adj.empty:
            st.error("No se pudo descargar historia para estos activos / rango.")
        else:
            wts = np.array([1 / adj.shape[1]] * adj.shape[1])
            port = stg.build_portfolio(adj, wts)
            st.caption(f"Cartera equiponderada de {adj.shape[1]} activos · "
                       f"{len(port)} días ({port.index.min()} → {port.index.max()}) · "
                       f"costo {cost_bps:.0f} bps/trade")

            def _equity_dd(df, dd, cum_strat_col, cum_bh_col, label,
                           buys=None, sells=None, buy_name="Long", sell_name="Short"):
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.72, 0.28], vertical_spacing=0.04)
                fig.add_trace(go.Scatter(x=df.index, y=(df[cum_bh_col] - 1) * 100,
                                         name="Buy & Hold", line=dict(color=CYAN, width=1.6)),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=(df[cum_strat_col] - 1) * 100,
                                         name=label, line=dict(color=ACCENT, width=2)),
                              row=1, col=1)
                if buys is not None and not buys.empty:
                    fig.add_trace(go.Scatter(x=buys.index, y=(buys[cum_strat_col] - 1) * 100,
                                             mode="markers", name=buy_name,
                                             marker=dict(symbol="triangle-up", color=GREEN, size=9)),
                                  row=1, col=1)
                if sells is not None and not sells.empty:
                    fig.add_trace(go.Scatter(x=sells.index, y=(sells[cum_strat_col] - 1) * 100,
                                             mode="markers", name=sell_name,
                                             marker=dict(symbol="triangle-down", color=RED, size=9)),
                                  row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=dd * 100, fill="tozeroy",
                                         line=dict(color=RED, width=1), name="Drawdown"),
                              row=2, col=1)
                fig.update_layout(**PLOTLY_LAYOUT, height=520)
                fig.update_yaxes(title_text="Retorno acum. (%)", row=1, col=1)
                fig.update_yaxes(title_text="DD %", row=2, col=1)
                return fig

            def _show_perf(perf):
                styled = perf.copy()
                for c in styled.columns:
                    styled[c] = styled[c].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
                st.dataframe(styled, use_container_width=True)

            def _mc_hist(mc, label):
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=(mc["rend_finales"] - 1) * 100, nbinsx=50,
                                           marker_color="#8B6F47", opacity=0.7))
                fig.add_vline(x=mc["real"] * 100, line=dict(color=GREEN, width=2, dash="dash"),
                              annotation_text="Real")
                fig.add_vline(x=mc["var_5"] * 100, line=dict(color=ACCENT, width=2, dash="dash"),
                              annotation_text="VaR 5%")
                fig.add_vline(x=mc["cvar_5"] * 100, line=dict(color=RED, width=2, dash="dash"),
                              annotation_text="CVaR 5%")
                fig.update_layout(**PLOTLY_LAYOUT, height=320,
                                  xaxis_title="Rendimiento final (%)", yaxis_title="Frecuencia")
                return fig

            def _run(fn, *a, **k):
                """Corre un backtest; si el período es muy corto, avisa y detiene el tab."""
                try:
                    return fn(*a, **k)
                except stg.InsufficientData as e:
                    st.warning(f"⏳ {e}")
                    st.stop()

            # ── TREND ───────────────────────────────────────────────────
            if strat == "Trend (SMA/LMA)":
                with st.spinner("Optimizando in-sample y evaluando out-of-sample..."):
                    r = _run(stg.backtest_sma_lma, port, cost=cost, is_frac=is_frac)
                ma1, ma2 = r["params"]
                k1, k2, k3 = st.columns(3)
                k1.metric("SMA / LMA óptimos", f"{ma1} / {ma2}")
                k2.metric("CAGR OOS", f"{r['cagr_oos'] * 100:+.2f}%")
                k3.metric("Overfitting Gap", f"{r['gap'] * 100:+.1f} pp",
                          help="CAGR_IS − CAGR_OOS. >15pp sugiere overfitting.")
                _show_perf(r["perf"])
                st.plotly_chart(_equity_dd(r["df"], r["dd"], "CUM_STRATEGY", "CUM_BH",
                                           r["label"], r["buys"], r["sells"],
                                           "Golden cross", "Death cross"),
                                use_container_width=True, config=PLOTLY_CONFIG)
                st.markdown("**RIESGO SIMULADO — MONTE CARLO (1000 trayectorias)**")
                st.plotly_chart(_mc_hist(r["mc"], r["label"]),
                                use_container_width=True, config=PLOTLY_CONFIG)

            # ── RSI ─────────────────────────────────────────────────────
            elif strat == "Mean Reversion (RSI)":
                cA, cB, cC = st.columns(3)
                rsi_win = cA.number_input("Ventana RSI", 5, 50, 14, key="rsi_w")
                osold = cB.number_input("Sobreventa", 10, 45, 30, key="rsi_os")
                obought = cC.number_input("Sobrecompra", 55, 90, 70, key="rsi_ob")
                with st.spinner("Backtest RSI..."):
                    r = _run(stg.backtest_rsi, port, window=int(rsi_win), oversold=int(osold),
                             overbought=int(obought), cost=cost, is_frac=is_frac)
                k1, k2, k3 = st.columns(3)
                k1.metric("Operaciones", r["n_trades"])
                k2.metric("Días en mercado", f"{r['days_in']}/{r['n_total']}")
                k3.metric("CAGR", f"{r['perf'].iloc[0]['CAGR %']:+.2f}%")
                _show_perf(r["perf"])
                fig = _equity_dd(r["df"], r["dd"], "CUM_STRATEGY", "CUM_BH", r["label"],
                                 r["buys"], r["sells"], "Compra RSI<os", "Venta RSI>ob")
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                st.markdown("**RSI**")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=r["df"].index, y=r["df"]["RSI"],
                                             line=dict(color=PURPLE, width=1.2), name="RSI"))
                fig_rsi.add_hline(y=obought, line=dict(color=RED, dash="dash"))
                fig_rsi.add_hline(y=osold, line=dict(color=GREEN, dash="dash"))
                fig_rsi.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_range=[0, 100])
                st.plotly_chart(fig_rsi, use_container_width=True, config=PLOTLY_CONFIG)
                st.plotly_chart(_mc_hist(r["mc"], r["label"]),
                                use_container_width=True, config=PLOTLY_CONFIG)

            # ── MOMENTUM ────────────────────────────────────────────────
            elif strat == "Momentum":
                mwin = st.slider("Ventana de momentum (días hábiles)", 21, 252, 126, key="mom_w")
                with st.spinner("Backtest Momentum..."):
                    r = _run(stg.backtest_momentum, port, window=int(mwin), cost=cost, is_frac=is_frac)
                k1, k2 = st.columns(2)
                k1.metric("Cambios de señal", r["n_trades"])
                k2.metric("CAGR", f"{r['perf'].iloc[0]['CAGR %']:+.2f}%")
                _show_perf(r["perf"])
                st.plotly_chart(_equity_dd(r["df"], r["dd"], "CUM_STRATEGY", "CUM_BH",
                                           r["label"], r["buys"], r["sells"], "Long", "Short"),
                                use_container_width=True, config=PLOTLY_CONFIG)
                st.markdown("**MOMENTUM CRUDO**")
                fig_m = go.Figure()
                fig_m.add_trace(go.Scatter(x=r["df"].index, y=r["df"]["MOMENTUM"] * 100,
                                           line=dict(color=ACCENT, width=1.2), name="Momentum"))
                fig_m.add_hline(y=0, line=dict(color="#888", dash="dash"))
                fig_m.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title="Momentum (%)")
                st.plotly_chart(fig_m, use_container_width=True, config=PLOTLY_CONFIG)
                st.plotly_chart(_mc_hist(r["mc"], r["label"]),
                                use_container_width=True, config=PLOTLY_CONFIG)

            # ── REBALANCEO ──────────────────────────────────────────────
            elif strat == "Rebalanceo":
                freq_label = st.selectbox("Frecuencia de rebalanceo",
                                          ["Mensual (ME)", "Trimestral (QE)", "Anual (YE)"],
                                          key="reb_freq")
                freq = {"Mensual (ME)": "ME", "Trimestral (QE)": "QE", "Anual (YE)": "YE"}[freq_label]
                with st.spinner("Simulando rebalanceo..."):
                    r = _run(stg.backtest_rebalanceo, adj, wts, freq=freq, cost=cost, is_frac=is_frac)
                k1, k2 = st.columns(2)
                k1.metric("Rebalanceos ejecutados", r["n_rebal"])
                k2.metric("Costos acumulados", f"{r['cost_acum'] / r['capital'] * 100:.3f}%")
                _show_perf(r["perf"])
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.72, 0.28], vertical_spacing=0.04)
                fig.add_trace(go.Scatter(x=r["df"].index, y=(r["df"]["BuyHold"] - 1) * 100,
                                         name="Buy & Hold", line=dict(color=CYAN, width=1.6)), row=1, col=1)
                fig.add_trace(go.Scatter(x=r["df"].index, y=(r["df"]["Rebalanceo"] - 1) * 100,
                                         name=r["label"], line=dict(color=ACCENT, width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=r["df"].index, y=r["dd_bh"] * 100, fill="tozeroy",
                                         line=dict(color=CYAN, width=1), name="DD B&H", opacity=0.5), row=2, col=1)
                fig.add_trace(go.Scatter(x=r["df"].index, y=r["dd_reb"] * 100, fill="tozeroy",
                                         line=dict(color=ACCENT, width=1), name="DD Rebal"), row=2, col=1)
                fig.update_layout(**PLOTLY_LAYOUT, height=520)
                fig.update_yaxes(title_text="Retorno acum. (%)", row=1, col=1)
                fig.update_yaxes(title_text="DD %", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                st.markdown("**PORTFOLIO DRIFT — pesos sin rebalancear (Buy & Hold)**")
                fig_dr = go.Figure()
                for col in r["bh_pesos"].columns:
                    fig_dr.add_trace(go.Scatter(x=r["bh_pesos"].index, y=r["bh_pesos"][col],
                                                mode="lines", stackgroup="one", name=col))
                fig_dr.update_layout(**PLOTLY_LAYOUT, height=320, yaxis_title="Peso",
                                     yaxis_range=[0, 1])
                st.plotly_chart(fig_dr, use_container_width=True, config=PLOTLY_CONFIG)

            # ── MACHINE LEARNING ────────────────────────────────────────
            elif strat == "Machine Learning":
                ml_mode = st.radio(
                    "Modo de ML",
                    ["Timing (señal on/off, pesos fijos)",
                     "Pesos dinámicos (rotación cross-sectional)"],
                    horizontal=True, key="ml_mode",
                    help="Timing: una señal 0/1 sobre toda la cartera. "
                         "Pesos dinámicos: un modelo POR ACCIÓN decide qué tener y con qué peso.")

                # ═══ MODO 1 · TIMING (on/off) ═══════════════════════════════
                if ml_mode.startswith("Timing"):
                    cA, cB = st.columns(2)
                    model_choice = cA.selectbox("Modelo", stg.ML_MODELS, index=1, key="ml_model")
                    wf_win = cB.slider("Ventana Walk-Forward (días)", 60, 504, 252, step=21, key="ml_win")
                    st.caption("Reentrena el modelo cada día con los últimos N días y predice el "
                               "siguiente; opera la cartera entera o va a cash. XGBoost/SVM tardan "
                               "~1 min la primera vez (después queda cacheado).")
                    if st.button("▶ CORRER WALK-FORWARD", key="ml_run", use_container_width=True):
                        st.session_state["_ml_go"] = True
                    if st.session_state.get("_ml_go"):
                        with st.spinner(f"Entrenando {model_choice} en walk-forward…"):
                            r = _run(run_ml_backtest, tuple(lab_tickers),
                                     lab_start_eff.strftime("%Y-%m-%d"),
                                     lab_end_eff.strftime("%Y-%m-%d"),
                                     model_choice, int(wf_win), cost)
                        if r is None:
                            st.error("No hay datos suficientes.")
                        else:
                            k1, k2, k3, k4 = st.columns(4)
                            k1.metric("Accuracy", f"{r['acc']:.3f}")
                            k2.metric("CAGR ML", f"{r['perf'].iloc[0]['CAGR %']:+.2f}%")
                            k3.metric("Cambios posición", r["n_trades"])
                            k4.metric("Días invertido", f"{r['days_in']}/{r['n_total']}")
                            _show_perf(r["perf"])
                            st.plotly_chart(_equity_dd(r["df"], r["dd"], "CUM_ML", "CUM_BH", r["label"]),
                                            use_container_width=True, config=PLOTLY_CONFIG)
                            cc1, cc2 = st.columns(2)
                            with cc1:
                                st.markdown("**MATRIZ DE CONFUSIÓN**")
                                cm = r["cm"]
                                fig_cm = go.Figure(go.Heatmap(
                                    z=cm, x=["Pred. Bajada", "Pred. Subida"],
                                    y=["Real Bajada", "Real Subida"], colorscale="Blues",
                                    text=cm, texttemplate="%{text}", showscale=False))
                                fig_cm.update_layout(**PLOTLY_LAYOUT, height=340)
                                st.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG)
                            with cc2:
                                if r["importances"] is not None:
                                    st.markdown("**IMPORTANCIA DE FEATURES**")
                                    imp = r["importances"]
                                    fig_i = go.Figure(go.Bar(x=imp.values, y=list(imp.index),
                                                             orientation="h", marker_color=GREEN))
                                    fig_i.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Importancia")
                                    st.plotly_chart(fig_i, use_container_width=True, config=PLOTLY_CONFIG)
                                else:
                                    st.info("SVM no expone feature_importances_.")
                            st.markdown("**RIESGO SIMULADO — MONTE CARLO**")
                            st.plotly_chart(_mc_hist(r["mc"], r["label"]),
                                            use_container_width=True, config=PLOTLY_CONFIG)
                    else:
                        st.info("Configurá el modelo y la ventana, y tocá **CORRER WALK-FORWARD**.")

                # ═══ MODO 2 · PESOS DINÁMICOS (rotación) ════════════════════
                else:
                    if len(lab_tickers) < 2:
                        st.warning("Este modo necesita al menos 2 activos para rotar entre ellos.")
                    else:
                        wc1, wc2, wc3, wc4 = st.columns(4)
                        w_model = wc1.selectbox("Modelo por activo", stg.ML_WEIGHT_MODELS,
                                                index=1, key="mlw_model")
                        wf_win = wc2.slider("Ventana WF (días)", 60, 504, 252, step=21, key="mlw_win")
                        umbral = wc3.slider("Umbral P(sube)", 0.50, 0.70, 0.50, step=0.01, key="mlw_umb")
                        max_peso = wc4.slider("Cap por activo", 0.20, 1.0, 0.40, step=0.05, key="mlw_cap")
                        st.caption("Entrena un modelo POR ACCIÓN cada día → P(sube). Arma la cartera por "
                                   "convicción (∝ p−0.5), con cap por activo, y rota día a día. Costo por "
                                   "turnover real. Es lo más pesado (N acciones × N días de modelos).")
                        if st.button("▶ CORRER ROTACIÓN ML", key="mlw_run", use_container_width=True):
                            st.session_state["_mlw_go"] = True
                        if st.session_state.get("_mlw_go"):
                            with st.spinner(f"Entrenando {w_model} por activo en walk-forward… "
                                            "(puede tardar 1-3 min; quedará cacheado)"):
                                r = _run(run_ml_weights_backtest, tuple(lab_tickers),
                                         lab_start_eff.strftime("%Y-%m-%d"),
                                         lab_end_eff.strftime("%Y-%m-%d"),
                                         w_model, int(wf_win), cost, float(umbral),
                                         float(max_peso))
                            if r is None:
                                st.error("No hay datos suficientes.")
                            else:
                                k1, k2, k3, k4 = st.columns(4)
                                k1.metric("CAGR", f"{r['perf'].iloc[0]['CAGR %']:+.2f}%")
                                k2.metric("Sharpe", f"{r['perf'].iloc[0]['Sharpe']:.2f}")
                                k3.metric("Turnover diario", f"{r['turnover_medio']*100:.0f}%")
                                k4.metric("% invertido medio", f"{r['invertido_medio']*100:.0f}%")
                                _show_perf(r["perf"])
                                st.plotly_chart(_equity_dd(r["df"], r["dd"], "CUM_ML", "CUM_BH", r["label"]),
                                                use_container_width=True, config=PLOTLY_CONFIG)
                                st.markdown("**COMPOSICIÓN DE LA CARTERA EN EL TIEMPO · rotación guiada por ML**")
                                pf = r["pesos_df"]
                                fig_w = go.Figure()
                                for col in pf.columns:
                                    fig_w.add_trace(go.Scatter(
                                        x=pf.index, y=pf[col] * 100, mode="lines",
                                        stackgroup="one", name=col, line=dict(width=0.5)))
                                fig_w.add_trace(go.Scatter(
                                    x=r["df"].index, y=r["df"]["Invertido"] * 100, mode="lines",
                                    name="% invertido", line=dict(color="#fff", width=1, dash="dot")))
                                fig_w.update_layout(**PLOTLY_LAYOUT, height=420,
                                                    yaxis_title="Peso (%)", yaxis_range=[0, 100])
                                st.plotly_chart(fig_w, use_container_width=True, config=PLOTLY_CONFIG)
                                cc1, cc2 = st.columns([1, 1.4])
                                with cc1:
                                    st.markdown("**PESO MEDIO POR ACTIVO (en cartera)**")
                                    pm = r["peso_medio"].sort_values(ascending=False)
                                    st.dataframe(pm.rename("Peso medio %").to_frame(),
                                                 use_container_width=True)
                                    st.caption(f"Costo total acumulado: {r['costo_total']*100:.2f}% del capital.")
                                with cc2:
                                    st.markdown("**RIESGO SIMULADO — MONTE CARLO**")
                                    st.plotly_chart(_mc_hist(r["mc"], r["label"]),
                                                    use_container_width=True, config=PLOTLY_CONFIG)
                        else:
                            st.info("Configurá los parámetros y tocá **CORRER ROTACIÓN ML**.")


# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    f"ACI I TERMINAL  ·  SESSION {datetime.now().strftime('%H:%M:%S')}  ·  "
    f"LIVE STREAM {GB_POLL_SEC}s · ALWAYS ON"
)
