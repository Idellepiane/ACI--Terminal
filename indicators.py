"""
indicators.py — Biblioteca de indicadores técnicos.

Consolida los 14 scripts de Clase 2 en funciones puras (data in → data out).
Cada función recibe un DataFrame con columnas estándar OHLCV (yahooquery format)
y devuelve el mismo DataFrame con las columnas del indicador añadidas.

Las versiones aquí son las "corregidas" de cada script (Wilder para RSI,
fórmula original de Keltner, OAD de Williams, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from sklearn.linear_model import LinearRegression


# ─────────────────────────────────────────────
# BANDAS DE BOLLINGER
# ─────────────────────────────────────────────
def bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    df = df.copy()
    df["MA"] = df["close"].rolling(window).mean()
    rolling_std = df["close"].rolling(window).std()
    df["Upper"] = df["MA"] + num_std * rolling_std
    df["Lower"] = df["MA"] - num_std * rolling_std
    return df


# ─────────────────────────────────────────────
# BANDAS DE KELTNER (fórmula original Chester Keltner, 1960)
# ─────────────────────────────────────────────
def keltner(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rango = df["high"] - df["low"]
    central = tp.rolling(window=period).mean()
    ancho = rango.rolling(window=period).mean()
    df["KC_Central"] = central
    df["KC_Upper"] = central + ancho
    df["KC_Lower"] = central - ancho
    return df


# ─────────────────────────────────────────────
# RSI (suavizado de Wilder)
# ─────────────────────────────────────────────
def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series = rsi_series.where(avg_loss != 0, 100)
    df["RSI"] = rsi_series
    return df


# ─────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────
def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    df["EMA_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    return df


# ─────────────────────────────────────────────
# COMMODITY CHANNEL INDEX (CCI)
# ─────────────────────────────────────────────
def cci(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (tp - sma_tp) / (0.015 * mad)
    return df


# ─────────────────────────────────────────────
# ESTOCÁSTICO (%K, %D)
# ─────────────────────────────────────────────
def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    df = df.copy()
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    df["%K"] = 100 * ((df["close"] - low_min) / (high_max - low_min))
    df["%D"] = df["%K"].rolling(window=d_period).mean()
    return df


# ─────────────────────────────────────────────
# MONEY FLOW INDEX
# ─────────────────────────────────────────────
def mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = tp * df["volume"]
    tp_diff = tp.diff()
    pos_mf = pd.Series(np.where(tp_diff > 0, raw_mf, 0.0), index=df.index)
    neg_mf = pd.Series(np.where(tp_diff < 0, raw_mf, 0.0), index=df.index)
    pos_sum = pos_mf.rolling(window=period).sum()
    neg_sum = neg_mf.rolling(window=period).sum()
    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi_series = 100 - (100 / (1 + money_ratio))
    df["MFI"] = mfi_series.where(neg_sum != 0, 100)
    return df


# ─────────────────────────────────────────────
# ON BALANCE VOLUME (vectorizado)
# ─────────────────────────────────────────────
def obv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    direction = np.sign(df["close"].diff()).fillna(0)
    df["OBV"] = (direction * df["volume"]).cumsum()
    return df


# ─────────────────────────────────────────────
# AROON (Up, Down, Oscillator)
# ─────────────────────────────────────────────
def aroon(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    df = df.copy().sort_index()
    aroon_up = df["high"].rolling(window + 1).apply(
        lambda x: 100 * np.argmax(x) / window, raw=True
    )
    aroon_down = df["low"].rolling(window + 1).apply(
        lambda x: 100 * np.argmin(x) / window, raw=True
    )
    df["Aroon Up"] = aroon_up
    df["Aroon Down"] = aroon_down
    df["Aroon Oscillator"] = aroon_up - aroon_down
    return df


# ─────────────────────────────────────────────
# OAD — OSCILADOR DE ACUMULACIÓN (Williams)
# ─────────────────────────────────────────────
def oad(df: pd.DataFrame, signal_window: int = 5) -> pd.DataFrame:
    df = df.copy()
    rango = df["high"] - df["low"]
    rango_seguro = rango.replace(0, np.nan)
    df["OAD"] = ((df["high"] - df["open"]) + (df["close"] - df["low"])) / (2 * rango_seguro) * 100
    df["OAD"] = df["OAD"].fillna(50)
    df["OAD Signal"] = df["OAD"].rolling(signal_window).mean()
    return df


# ─────────────────────────────────────────────
# MOMENTUM / IMPULSO
# ─────────────────────────────────────────────
def momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["Momentum"] = df["close"] - df["close"].shift(period)
    return df


# ─────────────────────────────────────────────
# EXPONENTE DE HURST (rolling)
# ─────────────────────────────────────────────
def _hurst_single(ts: np.ndarray) -> float:
    ts = np.asarray(ts, dtype=float)
    N = len(ts)
    if N < 20:
        return np.nan
    lags = np.arange(2, max(3, N // 2))
    tau = np.array([np.std(ts[lag:] - ts[:-lag]) for lag in lags])
    valid = tau > 0
    if valid.sum() < 3:
        return np.nan
    coefs = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)
    return float(coefs[0])


def hurst(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    df = df.copy()
    df["Hurst"] = df["close"].rolling(window=window).apply(_hurst_single, raw=True)
    return df


# ─────────────────────────────────────────────
# BANDAS DE AUTO-REGRESIÓN (regresión lineal ± kσ)
# ─────────────────────────────────────────────
def linreg_bands(df: pd.DataFrame, k: float = 2) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    X = np.arange(n).reshape(-1, 1).astype(float)
    y = df["close"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    sigma = (y - pred).std(ddof=1)
    df["LR_Recta"] = pred
    df["LR_Upper"] = pred + k * sigma
    df["LR_Lower"] = pred - k * sigma
    return df


# ─────────────────────────────────────────────
# ONDAS DE ELLIOTT — detección + etiquetado 1-5 / A-B-C
# ─────────────────────────────────────────────
def elliott(
    df: pd.DataFrame,
    window_length: int = 21,
    polyorder: int = 3,
    prominencia: float = 4.5,
    distancia: int = 15,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Retorna (df con columna 'Suavizado', índices de picos, índices de valles).
    """
    df = df.copy()
    close = df["close"].values

    if window_length > len(close):
        window_length = len(close)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder:
        suavizado = close
    else:
        suavizado = savgol_filter(close, window_length, polyorder)

    df["Suavizado"] = suavizado
    picos, _ = find_peaks(suavizado, prominence=prominencia, distance=distancia)
    valles, _ = find_peaks(-suavizado, prominence=prominencia, distance=distancia)
    return df, picos, valles


def label_elliott_waves(
    picos: np.ndarray,
    valles: np.ndarray,
    prices: np.ndarray,
) -> list[dict]:
    """
    Asigna etiquetas de Onda de Elliott (1, 2, 3, 4, 5, A, B, C) a cada pivote
    detectado, recorriéndolos en orden cronológico.

    Lógica:
      - Onda impulsiva: 5 sub-ondas (1, 3, 5 en dirección de la tendencia;
        2, 4 son retrocesos contra-tendencia dentro del impulso).
      - Onda correctiva: 3 sub-ondas (A, B, C) que siguen al impulso.
      - Después de A-B-C se reinicia el ciclo (se asume que viene otro impulso).

    Validaciones (suaves, marca con 'valid' = False si falla):
      - La onda 2 no debe retroceder por debajo del inicio de la onda 1.
      - La onda 3 no debe ser la más corta entre 1, 3, 5.
      - La onda 4 no debe solaparse con el rango de la onda 1.

    Returns:
        Lista de dicts: [{idx, kind ('peak'|'valley'), price, label, valid}]
    """
    pivots = (
        [(int(i), "peak", float(prices[int(i)])) for i in picos]
        + [(int(i), "valley", float(prices[int(i)])) for i in valles]
    )
    pivots.sort(key=lambda x: x[0])

    if not pivots:
        return []

    cycle = ["1", "2", "3", "4", "5", "A", "B", "C"]
    labeled: list[dict] = []
    for n, (idx, kind, price) in enumerate(pivots):
        labeled.append({
            "idx": idx,
            "kind": kind,
            "price": price,
            "label": cycle[n % len(cycle)],
            "valid": True,
        })

    # Validaciones por ciclo de 8 ondas
    for start in range(0, len(labeled), len(cycle)):
        block = labeled[start : start + len(cycle)]
        if len(block) < 5:
            continue
        w1, w2, w3, w4, w5 = block[:5]

        # 1) Onda 2 no debe retroceder por debajo del inicio de la 1.
        #    El "inicio de la 1" es el pivote anterior (si existe); si no,
        #    usamos el primer precio detectado.
        anchor_price = labeled[start - 1]["price"] if start > 0 else block[0]["price"]
        # En un impulso alcista: w2 (valley) > anchor
        # En uno bajista: w2 (peak) < anchor
        if w1["kind"] == "peak":  # impulso bajista (1=pico, 2=valle, etc.)
            if w2["price"] > anchor_price:
                w2["valid"] = False
        else:  # impulso alcista
            if w2["price"] < anchor_price:
                w2["valid"] = False

        # 2) Onda 3 no debe ser la más corta de 1, 3, 5 (medimos magnitud).
        d1 = abs(w1["price"] - anchor_price)
        d3 = abs(w3["price"] - w2["price"])
        d5 = abs(w5["price"] - w4["price"])
        if d3 < d1 and d3 < d5:
            w3["valid"] = False

        # 3) Onda 4 no debe solaparse con el rango de la onda 1
        #    (regla más estricta de Elliott).
        lo1, hi1 = sorted([anchor_price, w1["price"]])
        if lo1 <= w4["price"] <= hi1:
            w4["valid"] = False

    return labeled


# ─────────────────────────────────────────────
# REGISTRO de indicadores (para el dashboard)
# ─────────────────────────────────────────────
INDICATORS = {
    "Bandas de Bollinger": {
        "fn": bollinger,
        "kind": "overlay",
        "params": {"window": 20, "num_std": 2},
    },
    "Bandas de Keltner": {
        "fn": keltner,
        "kind": "overlay",
        "params": {"period": 10},
    },
    "RSI (Wilder)": {
        "fn": rsi,
        "kind": "oscillator",
        "params": {"period": 14},
    },
    "MACD": {
        "fn": macd,
        "kind": "oscillator",
        "params": {"fast": 12, "slow": 26, "signal": 9},
    },
    "CCI": {
        "fn": cci,
        "kind": "oscillator",
        "params": {"window": 20},
    },
    "Estocástico (%K / %D)": {
        "fn": stochastic,
        "kind": "oscillator",
        "params": {"k_period": 14, "d_period": 3},
    },
    "Money Flow Index": {
        "fn": mfi,
        "kind": "oscillator",
        "params": {"period": 14},
    },
    "On Balance Volume": {
        "fn": obv,
        "kind": "oscillator",
        "params": {},
    },
    "Aroon": {
        "fn": aroon,
        "kind": "oscillator",
        "params": {"window": 25},
    },
    "OAD (Williams)": {
        "fn": oad,
        "kind": "oscillator",
        "params": {"signal_window": 5},
    },
    "Momentum / Impulso": {
        "fn": momentum,
        "kind": "oscillator",
        "params": {"period": 10},
    },
    "Hurst (rolling)": {
        "fn": hurst,
        "kind": "oscillator",
        "params": {"window": 100},
    },
    "Bandas de Auto-Regresión": {
        "fn": linreg_bands,
        "kind": "overlay",
        "params": {"k": 2},
    },
    "Ondas de Elliott": {
        "fn": elliott,
        "kind": "elliott",
        "params": {
            "window_length": 21,
            "polyorder": 3,
            "prominencia": 4.5,
            "distancia": 15,
        },
    },
}
