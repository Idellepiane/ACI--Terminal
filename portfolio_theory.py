"""
portfolio_theory.py — Teoría de portfolio (Clase 3).

Consolida los 10 scripts de Clase 3 en funciones puras (data in → data out).
Sin matplotlib ni dependencias de UI — sólo numpy / pandas / scipy.

Cubre:
  · Matriz Σ de varianzas-covarianzas (daily / annual)
  · Correlación de Pearson, Spearman, Kendall
  · Distancia de Frobenius entre matrices de correlación
  · CAPM, β, Security Market Line, α de Jensen
  · Frontera eficiente vía Monte Carlo
  · Frontera eficiente vía optimización (SLSQP)
  · Portafolio tangente y Capital Market Line
  · Contribución al riesgo
  · Frontera con vistas propias (expected returns custom) con / sin bounds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


TRADING_DAYS_PER_YEAR = 252


# ════════════════════════════════════════════════════════════════════════════════
# RENDIMIENTOS
# ════════════════════════════════════════════════════════════════════════════════
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """log(P_t / P_{t-1}). Aditivos en el tiempo, suelen ser más simétricos."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """P_t / P_{t-1} − 1. Usados típicamente en CAPM y Markowitz."""
    return prices.pct_change(fill_method=None).dropna(how="all")


def annualize_return_from_daily(daily_returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    """(1 + r̄_d)^252 − 1"""
    return (1 + daily_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1


def annualize_vol_from_daily(daily_returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    """σ_d · √252"""
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


# ════════════════════════════════════════════════════════════════════════════════
# COVARIANZA Y CORRELACIÓN
# ════════════════════════════════════════════════════════════════════════════════
def cov_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Matriz Σ de varianzas-covarianzas.
    Si annualize=True multiplica por 252 (independencia entre días hábiles).
    """
    cov = returns.cov()
    return cov * TRADING_DAYS_PER_YEAR if annualize else cov


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    method: 'pearson' (default), 'spearman' (rangos), 'kendall' (concordancia).
    """
    return returns.corr(method=method)


def correlation_kinds(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson, Spearman, Kendall sobre dos arrays 1D."""
    p, _ = stats.pearsonr(x, y)
    s, _ = stats.spearmanr(x, y)
    k, _ = stats.kendalltau(x, y)
    return {"pearson": float(p), "spearman": float(s), "kendall": float(k)}


def frobenius_distance(A: pd.DataFrame | np.ndarray, B: pd.DataFrame | np.ndarray) -> float:
    """‖A − B‖_F = √Σᵢⱼ(aᵢⱼ − bᵢⱼ)²"""
    return float(np.linalg.norm(np.asarray(A) - np.asarray(B), "fro"))


def split_correlations(returns: pd.DataFrame, n_periods: int = 2) -> list[pd.DataFrame]:
    """
    Divide el panel de retornos en n_periods sub-períodos de igual largo y
    devuelve la matriz de correlación de cada uno.
    """
    chunk = len(returns) // n_periods
    return [returns.iloc[i * chunk:(i + 1) * chunk].corr() for i in range(n_periods)]


# ════════════════════════════════════════════════════════════════════════════════
# CAPM / SML
# ════════════════════════════════════════════════════════════════════════════════
def capm_analysis(
    returns: pd.DataFrame,
    benchmark: str,
    rf: float = 0.04,
) -> pd.DataFrame:
    """
    Estima β, retorno CAPM y α para cada ticker contra el benchmark.

    Args:
        returns: DataFrame de retornos diarios. Una columna por ticker
                 (incluyendo el benchmark).
        benchmark: nombre de la columna que actúa de mercado.
        rf: tasa libre de riesgo anual.

    Returns:
        DataFrame con columnas:
          - 'annual_return'  : retorno anualizado realizado
          - 'annual_vol'     : volatilidad anualizada
          - 'beta'           : Cov(Ri, Rm) / Var(Rm) — daily
          - 'capm_return'    : R_f + β·(E[Rm] − R_f)
          - 'alpha'          : retorno realizado − CAPM
        Indexed por ticker (sin el benchmark).
    """
    if benchmark not in returns.columns:
        raise ValueError(f"Benchmark '{benchmark}' not in returns columns.")

    tickers = [c for c in returns.columns if c != benchmark]
    annual_ret = annualize_return_from_daily(returns)
    annual_vol = annualize_vol_from_daily(returns)

    # IMPORTANTE: pandas .cov() y .var() usan ambos ddof=1 → coherentes.
    cov = returns.cov()
    var_mkt = returns[benchmark].var()
    betas = cov.loc[tickers, benchmark] / var_mkt

    mkt_return = annual_ret[benchmark]
    capm_ret = rf + betas * (mkt_return - rf)
    alpha = annual_ret[tickers] - capm_ret

    return pd.DataFrame({
        "annual_return": annual_ret[tickers],
        "annual_vol": annual_vol[tickers],
        "beta": betas,
        "capm_return": capm_ret,
        "alpha": alpha,
    })


def sml_line(rf: float, mkt_return: float, betas: np.ndarray) -> np.ndarray:
    """E[R] = R_f + β · (E[Rm] − R_f) — para pintar la SML."""
    return rf + betas * (mkt_return - rf)


# ════════════════════════════════════════════════════════════════════════════════
# FRONTERA EFICIENTE  ·  MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════════
def monte_carlo_frontier(
    annual_return: pd.Series,
    annual_cov: pd.DataFrame,
    n_portfolios: int = 10_000,
    rf: float = 0.04,
    seed: int = 42,
) -> dict:
    """
    Genera n_portfolios carteras aleatorias (long-only, w sum=1) y devuelve:
      - 'results': array (3, n) con [retorno, vol, sharpe]
      - 'weights': lista de arrays con los pesos de cada cartera
      - 'tangent': dict con índice y métricas del óptimo Monte Carlo
      - 'min_vol': dict con índice y métricas del de mínima varianza
    """
    rng = np.random.default_rng(seed)
    n = len(annual_return)
    results = np.zeros((3, n_portfolios))
    weights_record = np.zeros((n_portfolios, n))

    cov_arr = annual_cov.values
    ret_arr = annual_return.values

    for i in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        weights_record[i] = w
        r = float(w @ ret_arr)
        v = float(np.sqrt(w @ cov_arr @ w))
        s = (r - rf) / v if v > 0 else 0.0
        results[:, i] = [r, v, s]

    idx_t = int(np.argmax(results[2]))
    idx_m = int(np.argmin(results[1]))

    return {
        "results": results,
        "weights": weights_record,
        "tickers": list(annual_return.index),
        "tangent": {
            "idx": idx_t,
            "weights": weights_record[idx_t],
            "return": results[0, idx_t],
            "vol": results[1, idx_t],
            "sharpe": results[2, idx_t],
        },
        "min_vol": {
            "idx": idx_m,
            "weights": weights_record[idx_m],
            "return": results[0, idx_m],
            "vol": results[1, idx_m],
        },
    }


# ════════════════════════════════════════════════════════════════════════════════
# FRONTERA EFICIENTE  ·  OPTIMIZACIÓN SLSQP
# ════════════════════════════════════════════════════════════════════════════════
def _port_stats(w: np.ndarray, annual_return: np.ndarray, annual_cov: np.ndarray):
    r = float(w @ annual_return)
    v = float(np.sqrt(w @ annual_cov @ w))
    return r, v


def efficient_frontier(
    annual_return: pd.Series,
    annual_cov: pd.DataFrame,
    n_points: int = 100,
    bound_min: float = 0.0,
    bound_max: float = 1.0,
) -> pd.DataFrame:
    """
    Resuelve, para n_points niveles de retorno objetivo, el portafolio de
    MÍNIMA varianza:
        min  w'Σw
        s.t. w'μ = target_r
             Σw = 1
             bound_min ≤ wᵢ ≤ bound_max

    Devuelve DataFrame con columnas: 'target_return', 'vol', 'success'.
    """
    n = len(annual_return)
    ret_arr = annual_return.values
    cov_arr = annual_cov.values
    bounds = tuple((bound_min, bound_max) for _ in range(n))
    w0 = np.ones(n) / n

    def vol(w):
        return float(np.sqrt(w @ cov_arr @ w))

    def ret(w):
        return float(w @ ret_arr)

    targets = np.linspace(annual_return.min(), annual_return.max(), n_points)
    rows = []
    for r_target in targets:
        cons = [
            {"type": "eq", "fun": lambda w, r=r_target: ret(w) - r},
            {"type": "eq", "fun": lambda w: w.sum() - 1},
        ]
        res = minimize(vol, w0, method="SLSQP", bounds=bounds, constraints=cons)
        rows.append({
            "target_return": r_target,
            "vol": res.fun if res.success else np.nan,
            "success": bool(res.success),
        })

    return pd.DataFrame(rows)


def tangent_portfolio(
    annual_return: pd.Series,
    annual_cov: pd.DataFrame,
    rf: float = 0.04,
    bound_min: float = 0.0,
    bound_max: float = 1.0,
) -> dict:
    """
    Resuelve max Sharpe → portafolio tangente:
        max  (w'μ − R_f) / √(w'Σw)
        s.t. Σw = 1
             bound_min ≤ wᵢ ≤ bound_max
    """
    n = len(annual_return)
    ret_arr = annual_return.values
    cov_arr = annual_cov.values
    bounds = tuple((bound_min, bound_max) for _ in range(n))
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        r, v = _port_stats(w, ret_arr, cov_arr)
        return -(r - rf) / v if v > 0 else 1e9

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    w_opt = res.x
    r, v = _port_stats(w_opt, ret_arr, cov_arr)

    return {
        "tickers": list(annual_return.index),
        "weights": w_opt,
        "return": r,
        "vol": v,
        "sharpe": (r - rf) / v if v > 0 else 0.0,
        "success": bool(res.success),
        "bounds": (bound_min, bound_max),
    }


def min_variance_portfolio(
    annual_cov: pd.DataFrame,
    bound_min: float = 0.0,
    bound_max: float = 1.0,
) -> dict:
    """Portafolio de mínima varianza global (sin objetivo de retorno)."""
    n = annual_cov.shape[0]
    cov_arr = annual_cov.values
    bounds = tuple((bound_min, bound_max) for _ in range(n))
    w0 = np.ones(n) / n

    def vol(w):
        return float(np.sqrt(w @ cov_arr @ w))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(vol, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return {
        "weights": res.x,
        "vol": res.fun,
        "tickers": list(annual_cov.index),
    }


# ════════════════════════════════════════════════════════════════════════════════
# CONTRIBUCIÓN AL RIESGO
# ════════════════════════════════════════════════════════════════════════════════
def risk_contribution(weights: np.ndarray, annual_cov: pd.DataFrame) -> pd.Series:
    """
    Contribución porcentual de cada activo al riesgo total del portafolio.

    contrib_i = wᵢ · (Σw)ᵢ / σ_p
    pct       = 100 · contrib_i / Σ contrib
    """
    cov_arr = annual_cov.values
    sigma_p = float(np.sqrt(weights @ cov_arr @ weights))
    if sigma_p <= 0:
        return pd.Series(np.zeros_like(weights), index=annual_cov.index)
    marg = cov_arr @ weights
    contrib = weights * marg / sigma_p
    pct = 100 * contrib / contrib.sum()
    return pd.Series(pct, index=annual_cov.index)


# ════════════════════════════════════════════════════════════════════════════════
# CAPITAL MARKET LINE
# ════════════════════════════════════════════════════════════════════════════════
def cml_points(rf: float, sharpe: float, sigma_max: float, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye los puntos (σ, E[R]) de la CML:
        E[R] = R_f + Sharpe · σ
    para σ ∈ [0, sigma_max].
    """
    sigmas = np.linspace(0, sigma_max, n)
    returns = rf + sharpe * sigmas
    return sigmas, returns
