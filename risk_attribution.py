"""
risk_attribution.py — Clase 4: VaR/CVaR y Performance Attribution.

Consolida los 6 scripts de Clase 4 en funciones puras (data in → data out),
sin matplotlib ni dependencias de UI. El dashboard (app.py) llama estas
funciones y dibuja con Plotly en el tema Bloomberg.

Cubre:
  · VaR y CVaR paramétricos (normal)           → 01_VaR_Parametrico.py
  · VaR y CVaR históricos (empírico)           → 02_VaR_Historico.py
  · VaR y CVaR Monte Carlo (normal multivar.)  → 03_VaR_MonteCarlo.py
  · Comparación de los 3 métodos               → 04_VaR_Comparacion.py
  · Brinson-Hood-Beebower (3 efectos)          → 05_PA_Brinson_Hood_Beebower.py
  · Brinson-Fachler (2 efectos)                → 06_PA_Brinson_Fachler.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ════════════════════════════════════════════════════════════════════════════════
# VaR / CVaR
# ════════════════════════════════════════════════════════════════════════════════
def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Retorno diario de la cartera = combinación lineal de retornos por activo."""
    return returns @ weights


def var_parametrico(returns: pd.DataFrame, weights: np.ndarray,
                    alpha: float = 0.95, capital: float = 1_000_000,
                    dias: int = 1) -> dict:
    """
    VaR y CVaR asumiendo retornos Normal(μ, σ) de la cartera.
        VaR_α  = −(μ_T + z·σ_T) · V
        CVaR_α = −(μ_T − σ_T·φ(z)/(1−α)) · V
    con z = Φ⁻¹(1−α), μ_T = μ·T, σ_T = σ·√T.
    """
    media_assets = returns.mean().values
    cov_assets = returns.cov().values
    mu = float(weights @ media_assets)
    sigma = float(np.sqrt(weights @ cov_assets @ weights))

    z = norm.ppf(1 - alpha)
    mu_T = mu * dias
    sigma_T = sigma * np.sqrt(dias)

    ret_var = mu_T + z * sigma_T
    ret_cvar = mu_T - sigma_T * norm.pdf(z) / (1 - alpha)

    return {
        "metodo": "Paramétrico",
        "var_pct": -ret_var, "cvar_pct": -ret_cvar,
        "var_usd": -ret_var * capital, "cvar_usd": -ret_cvar * capital,
        "ret_var": ret_var, "ret_cvar": ret_cvar,
        "mu_T": mu_T, "sigma_T": sigma_T, "z": z,
    }


def var_historico(returns: pd.DataFrame, weights: np.ndarray,
                  alpha: float = 0.95, capital: float = 1_000_000) -> dict:
    """
    VaR y CVaR empíricos: percentil α de los retornos observados y promedio
    de la cola. No supone normalidad.
    """
    ret_cartera = (returns @ weights).values
    N = len(ret_cartera)
    orden = np.sort(ret_cartera)
    idx = max(int((1 - alpha) * N), 1)
    ret_var = orden[idx - 1]
    ret_cvar = orden[:idx].mean()

    return {
        "metodo": "Histórico",
        "var_pct": -ret_var, "cvar_pct": -ret_cvar,
        "var_usd": -ret_var * capital, "cvar_usd": -ret_cvar * capital,
        "ret_var": ret_var, "ret_cvar": ret_cvar,
        "ret_cartera": ret_cartera, "n_obs": N, "obs_cola": idx,
        "peor_dia": float(orden[0]),
    }


def var_montecarlo(returns: pd.DataFrame, weights: np.ndarray,
                   alpha: float = 0.95, capital: float = 1_000_000,
                   dias: int = 1, num_sim: int = 10_000, seed: int = 42) -> dict:
    """
    VaR/CVaR simulando la distribución MULTIVARIADA de los activos
    (Normal(μ, Σ)) y agregando a nivel cartera — captura correlaciones.
    """
    rng = np.random.default_rng(seed)
    mu_assets = returns.mean().values * dias
    cov_assets = returns.cov().values * dias

    sim_assets = rng.multivariate_normal(mu_assets, cov_assets, num_sim)
    sim_cartera = sim_assets @ weights
    orden = np.sort(sim_cartera)
    idx = int((1 - alpha) * num_sim)
    ret_var = orden[idx]
    ret_cvar = orden[:idx].mean()

    return {
        "metodo": "Monte Carlo",
        "var_pct": -ret_var, "cvar_pct": -ret_cvar,
        "var_usd": -ret_var * capital, "cvar_usd": -ret_cvar * capital,
        "ret_var": ret_var, "ret_cvar": ret_cvar,
        "sim_cartera": sim_cartera, "num_sim": num_sim,
        "peor": float(orden[0]), "mejor": float(orden[-1]),
    }


def var_comparacion(returns: pd.DataFrame, weights: np.ndarray,
                    alpha: float = 0.95, capital: float = 1_000_000,
                    num_sim: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Tabla comparativa de los 3 métodos sobre la misma cartera."""
    p = var_parametrico(returns, weights, alpha, capital, dias=1)
    h = var_historico(returns, weights, alpha, capital)
    m = var_montecarlo(returns, weights, alpha, capital, dias=1,
                       num_sim=num_sim, seed=seed)
    return pd.DataFrame({
        "Método": [p["metodo"], h["metodo"], m["metodo"]],
        "VaR (USD)": [p["var_usd"], h["var_usd"], m["var_usd"]],
        "CVaR (USD)": [p["cvar_usd"], h["cvar_usd"], m["cvar_usd"]],
        "VaR (%)": [p["var_pct"], h["var_pct"], m["var_pct"]],
        "CVaR (%)": [p["cvar_pct"], h["cvar_pct"], m["cvar_pct"]],
    })


# ════════════════════════════════════════════════════════════════════════════════
# PERFORMANCE ATTRIBUTION  ·  Brinson
# ════════════════════════════════════════════════════════════════════════════════
def bhb_attribution(sectores: list[str], pesos_port: dict[str, float],
                    ret_port: dict[str, float],
                    pesos_bench: dict[str, float],
                    ret_bench: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """
    Brinson-Hood-Beebower (1986): Exceso = Asignación + Selección + Interacción.
        Asignación  = (w_P − w_B) · R_B
        Selección   =  w_B · (R_P − R_B)
        Interacción = (w_P − w_B) · (R_P − R_B)
    """
    df = pd.DataFrame({
        "Sector": sectores,
        "Weight_P": [pesos_port.get(s, 0.0) for s in sectores],
        "Return_P": [ret_port.get(s, 0.0) for s in sectores],
        "Weight_B": [pesos_bench.get(s, 0.0) for s in sectores],
        "Return_B": [ret_bench.get(s, 0.0) for s in sectores],
    })
    df["Asignacion"] = (df["Weight_P"] - df["Weight_B"]) * df["Return_B"]
    df["Seleccion"] = df["Weight_B"] * (df["Return_P"] - df["Return_B"])
    df["Interaccion"] = (df["Weight_P"] - df["Weight_B"]) * (df["Return_P"] - df["Return_B"])
    df["Total"] = df["Asignacion"] + df["Seleccion"] + df["Interaccion"]

    R_P = float((df["Weight_P"] * df["Return_P"]).sum())
    R_B = float((df["Weight_B"] * df["Return_B"]).sum())
    exceso = R_P - R_B
    totals = {
        "R_P": R_P, "R_B": R_B, "exceso": exceso,
        "asignacion": float(df["Asignacion"].sum()),
        "seleccion": float(df["Seleccion"].sum()),
        "interaccion": float(df["Interaccion"].sum()),
        "check": exceso - float(df[["Asignacion", "Seleccion", "Interaccion"]].sum().sum()),
    }
    return df, totals


def bf_attribution(sectores: list[str], pesos_port: dict[str, float],
                   ret_port: dict[str, float],
                   pesos_bench: dict[str, float],
                   ret_bench: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """
    Brinson-Fachler (1985): Exceso = Asignación + Selección (sin interacción).
        Asignación BF = (w_P − w_B) · (R_B − R̄_B)   ← resta benchmark total
        Selección  BF =  w_P · (R_P − R_B)           ← usa w_P (absorbe interacción)
    """
    df = pd.DataFrame({
        "Sector": sectores,
        "Weight_P": [pesos_port.get(s, 0.0) for s in sectores],
        "Return_P": [ret_port.get(s, 0.0) for s in sectores],
        "Weight_B": [pesos_bench.get(s, 0.0) for s in sectores],
        "Return_B": [ret_bench.get(s, 0.0) for s in sectores],
    })
    R_P = float((df["Weight_P"] * df["Return_P"]).sum())
    R_B = float((df["Weight_B"] * df["Return_B"]).sum())
    exceso = R_P - R_B

    df["Allocation"] = (df["Weight_P"] - df["Weight_B"]) * (df["Return_B"] - R_B)
    df["Selection"] = df["Weight_P"] * (df["Return_P"] - df["Return_B"])
    df["Total"] = df["Allocation"] + df["Selection"]

    totals = {
        "R_P": R_P, "R_B": R_B, "exceso": exceso,
        "allocation": float(df["Allocation"].sum()),
        "selection": float(df["Selection"].sum()),
        "check": exceso - float(df["Total"].sum()),
    }
    return df, totals
