"""
strategies.py — Clase 5: Backtesting & Machine Learning.

Consolida los 6 scripts de Clase 5 en funciones puras. Todo el pipeline de
backtest (reglas → ejecución con shift(1) → costos → métricas) más las
estrategias ML con Walk-Forward Analysis.

Estrategias:
  · Cruce de medias móviles (trend following)  → BT_Medias_Moviles.py
  · RSI (mean reversion, long/flat)            → BT_RSI.py
  · Momentum 6m (long/short)                   → RT_Momentum.py
  · Rebalanceo periódico vs Buy & Hold         → RT_Rebalanceo.py
  · ML Walk-Forward (Tree/RF/XGBoost/SVM)      → BT_Machine_Learning(_II).py

Convención: retornos LOG diarios salvo en rebalanceo (aritméticos, porque
suma ponderada de activos). Costos en bps por cambio de posición.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RF_ANUAL_DEFAULT = 0.045
TRADING_DAYS = 252


# ════════════════════════════════════════════════════════════════════════════════
# MÉTRICAS (Slide 28)
# ════════════════════════════════════════════════════════════════════════════════
def cagr(returns: pd.Series) -> float:
    """CAGR a partir de retornos LOG diarios (suma logs ⇒ producto)."""
    if len(returns) == 0:
        return np.nan
    cumulative = np.exp(returns.sum())
    years = len(returns) / TRADING_DAYS
    return cumulative ** (1 / years) - 1 if years > 0 else np.nan


def cagr_arit(returns: pd.Series) -> float:
    """CAGR sobre retornos ARITMÉTICOS (para el rebalanceo)."""
    if len(returns) == 0:
        return np.nan
    cumulative = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return cumulative ** (1 / years) - 1 if years > 0 else np.nan


def volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def downside_vol(returns: pd.Series, target: float = 0.0) -> float:
    down = returns[returns < target]
    return down.std() * np.sqrt(TRADING_DAYS) if len(down) else np.nan


def sharpe(returns: pd.Series, rf: float = RF_ANUAL_DEFAULT, arit: bool = False) -> float:
    vol = volatility(returns)
    c = cagr_arit(returns) if arit else cagr(returns)
    return (c - rf) / vol if vol and vol > 0 else np.nan


def sortino(returns: pd.Series, rf: float = RF_ANUAL_DEFAULT, arit: bool = False) -> float:
    dvol = downside_vol(returns)
    c = cagr_arit(returns) if arit else cagr(returns)
    return (c - rf) / dvol if dvol and dvol > 0 else np.nan


def max_drawdown(cum_returns: pd.Series) -> tuple[float, pd.Series]:
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    return drawdown.min(), drawdown


def calmar(returns: pd.Series, arit: bool = False) -> float:
    cum = (1 + returns).cumprod() if arit else np.exp(returns.cumsum())
    mdd, _ = max_drawdown(cum)
    c = cagr_arit(returns) if arit else cagr(returns)
    return c / abs(mdd) if mdd < 0 else np.nan


def hit_ratio(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    return (nonzero > 0).mean() if len(nonzero) else np.nan


def _perf_row(rets: pd.Series, cum: pd.Series, arit: bool = False) -> dict:
    mdd, _ = max_drawdown(cum)
    total = (cum.iloc[-1] - 1) if len(cum) else np.nan
    return {
        "CAGR %": (cagr_arit(rets) if arit else cagr(rets)) * 100,
        "Vol Anual %": volatility(rets) * 100,
        "Sharpe": sharpe(rets, arit=arit),
        "Sortino": sortino(rets, arit=arit),
        "Max DD %": mdd * 100,
        "Calmar": calmar(rets, arit=arit),
        "Hit Ratio %": hit_ratio(rets) * 100,
        "Retorno Total %": total * 100,
    }


def montecarlo_var(strategy_returns: pd.Series, cum_real: float,
                   num_sim: int = 1000, seed: int = 7) -> dict:
    """Bootstrap paramétrico Normal → VaR/CVaR 5% sobre el rendimiento final."""
    rng = np.random.default_rng(seed)
    sim = rng.normal(loc=strategy_returns.mean(), scale=strategy_returns.std(),
                     size=(len(strategy_returns), num_sim))
    rend_finales = (1 + pd.DataFrame(sim)).cumprod().iloc[-1, :]
    var_5 = np.percentile(rend_finales, 5)
    cvar_5 = rend_finales[rend_finales <= var_5].mean()
    return {
        "rend_finales": rend_finales.values,
        "var_5": var_5 - 1, "cvar_5": cvar_5 - 1, "real": cum_real - 1,
    }


# ════════════════════════════════════════════════════════════════════════════════
# PORTAFOLIO BASE
# ════════════════════════════════════════════════════════════════════════════════
def build_portfolio(adj_close: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Valor de la cartera equiponderada = precios · pesos."""
    return adj_close.dot(weights)


# ════════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA 1 · CRUCE DE MEDIAS MÓVILES (trend following, long/short)
# ════════════════════════════════════════════════════════════════════════════════
def _bt_sma_lma_total_return(df: pd.DataFrame, ma1: int, ma2: int, cost: float) -> float:
    d = df.copy()
    d["SMA"] = d["Portfolio"].rolling(ma1).mean()
    d["LMA"] = d["Portfolio"].rolling(ma2).mean()
    d = d.dropna()
    d["SIGNAL"] = np.where(d["SMA"] > d["LMA"], 1, -1)
    d["SIGNAL"] = d["SIGNAL"].shift(1)
    d["RETURN"] = np.log(d["Portfolio"]).diff().fillna(0)
    d["TRADES"] = (d["SIGNAL"].diff().abs() / 2).fillna(0)
    d["STRATEGY"] = d["SIGNAL"] * d["RETURN"] - d["TRADES"] * cost
    return np.exp(d["STRATEGY"].sum()) - 1


def backtest_sma_lma(portfolio: pd.Series, cost: float = 0.0015,
                     rf: float = RF_ANUAL_DEFAULT,
                     ma1_range=range(10, 100, 10),
                     ma2_range=range(100, 250, 10),
                     is_frac: float = 0.8) -> dict:
    """
    Optimiza (MA1, MA2) en in-sample (80%) y aplica los óptimos out-of-sample.
    Devuelve métricas OOS estrategia vs Buy&Hold, equity, drawdown y overfit gap.
    """
    df = pd.DataFrame({"Portfolio": portfolio.values}, index=portfolio.index)
    insample = int(len(df) * is_frac)
    df_train, df_test = df[:insample].copy(), df[insample:].copy()

    grid = []
    for ma1 in ma1_range:
        for ma2 in ma2_range:
            if ma1 < ma2:
                grid.append({"MA1": ma1, "MA2": ma2,
                             "ret": _bt_sma_lma_total_return(df_train, ma1, ma2, cost)})
    grid_df = pd.DataFrame(grid)
    best = grid_df.sort_values("ret", ascending=False).iloc[0]
    ma1_opt, ma2_opt = int(best["MA1"]), int(best["MA2"])

    d = df_test
    d["SMA"] = d["Portfolio"].rolling(ma1_opt).mean()
    d["LMA"] = d["Portfolio"].rolling(ma2_opt).mean()
    d = d.dropna()
    d["SIGNAL"] = np.where(d["SMA"] > d["LMA"], 1, -1)
    d["SIGNAL"] = d["SIGNAL"].shift(1)
    d["RETURN"] = np.log(d["Portfolio"]).diff().fillna(0)
    d["TRADES"] = (d["SIGNAL"].diff().abs() / 2).fillna(0)
    d["STRATEGY"] = d["SIGNAL"] * d["RETURN"] - d["TRADES"] * cost
    d["CUM_STRATEGY"] = np.exp(d["STRATEGY"].cumsum())
    d["CUM_BH"] = np.exp(d["RETURN"].cumsum())
    d = d.dropna(subset=["SIGNAL"])

    _, dd = max_drawdown(d["CUM_STRATEGY"])
    perf = pd.DataFrame(
        [_perf_row(d["STRATEGY"], d["CUM_STRATEGY"]),
         _perf_row(d["RETURN"], d["CUM_BH"])],
        index=[f"Trend SMA({ma1_opt})/LMA({ma2_opt})", "Buy & Hold"],
    ).round(2)

    cagr_is = (1 + best["ret"]) ** (TRADING_DAYS / len(df_train)) - 1
    gap = cagr_is - cagr(d["STRATEGY"])

    sig_chg = d["SIGNAL"].diff()
    buys = d[sig_chg == 2]
    sells = d[sig_chg == -2]

    return {
        "df": d, "dd": dd, "perf": perf, "params": (ma1_opt, ma2_opt),
        "cagr_is": cagr_is, "cagr_oos": cagr(d["STRATEGY"]), "gap": gap,
        "buys": buys, "sells": sells, "grid": grid_df,
        "mc": montecarlo_var(d["STRATEGY"], d["CUM_STRATEGY"].iloc[-1]),
        "label": f"Trend SMA({ma1_opt})/LMA({ma2_opt})",
    }


# ════════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA 2 · RSI (mean reversion, long/flat)
# ════════════════════════════════════════════════════════════════════════════════
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def backtest_rsi(portfolio: pd.Series, window: int = 14,
                 oversold: int = 30, overbought: int = 70,
                 cost: float = 0.0015, rf: float = RF_ANUAL_DEFAULT,
                 is_frac: float = 0.8) -> dict:
    df = pd.DataFrame({"Portfolio": portfolio.values}, index=portfolio.index)
    insample = int(len(df) * is_frac)
    d = df[insample:].dropna().copy()
    d["RSI"] = compute_rsi(d["Portfolio"], window=window)

    position, signals = 0, []
    for rsi in d["RSI"]:
        if position == 0 and rsi < oversold:
            sig, position = 1, 1
        elif position == 1 and rsi > overbought:
            sig, position = -1, 0
        else:
            sig = 0
        signals.append(sig)
    d["SIGNAL_RAW"] = signals
    d["SIGNAL"] = d["SIGNAL_RAW"].shift(1)
    d = d.dropna(subset=["SIGNAL"])

    d["POSITION"] = d["SIGNAL"].replace(0, np.nan).ffill().fillna(0)
    d.loc[d["POSITION"] == -1, "POSITION"] = 0
    d["RETURN"] = np.log(d["Portfolio"]).diff().fillna(0)
    d["TRADES"] = d["POSITION"].diff().abs().fillna(0)
    d["STRATEGY"] = d["POSITION"] * d["RETURN"] - d["TRADES"] * cost
    d["CUM_STRATEGY"] = np.exp(d["STRATEGY"].cumsum())
    d["CUM_BH"] = np.exp(d["RETURN"].cumsum())

    _, dd = max_drawdown(d["CUM_STRATEGY"])
    perf = pd.DataFrame(
        [_perf_row(d["STRATEGY"], d["CUM_STRATEGY"]),
         _perf_row(d["RETURN"], d["CUM_BH"])],
        index=["RSI (long/flat)", "Buy & Hold"],
    ).round(2)

    n_trades = int(d["TRADES"].sum())
    days_in = int(d["POSITION"].sum())
    return {
        "df": d, "dd": dd, "perf": perf,
        "n_trades": n_trades, "days_in": days_in, "n_total": len(d),
        "oversold": oversold, "overbought": overbought,
        "buys": d[d["SIGNAL_RAW"] == 1], "sells": d[d["SIGNAL_RAW"] == -1],
        "mc": montecarlo_var(d["STRATEGY"], d["CUM_STRATEGY"].iloc[-1]),
        "label": f"RSI {window} ({oversold}/{overbought})",
    }


# ════════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA 3 · MOMENTUM (long/short)
# ════════════════════════════════════════════════════════════════════════════════
def backtest_momentum(portfolio: pd.Series, window: int = 126,
                      cost: float = 0.0015, rf: float = RF_ANUAL_DEFAULT,
                      is_frac: float = 0.8) -> dict:
    df = pd.DataFrame({"Portfolio": portfolio.values}, index=portfolio.index)
    insample = int(len(df) * is_frac)
    d = df[insample:].copy()
    d["MOMENTUM"] = d["Portfolio"].pct_change(window)
    d["SIGNAL_RAW"] = 0
    d.loc[d["MOMENTUM"] > 0, "SIGNAL_RAW"] = 1
    d.loc[d["MOMENTUM"] < 0, "SIGNAL_RAW"] = -1
    d["SIGNAL"] = d["SIGNAL_RAW"].shift(1)
    d = d.dropna(subset=["SIGNAL"])

    d["RETURN"] = np.log(d["Portfolio"]).diff().fillna(0)
    d["POSITION"] = d["SIGNAL"].replace(0, np.nan).ffill().fillna(0)
    d["TRADES"] = (d["POSITION"].diff().abs() / 2).fillna(0)
    d["STRATEGY"] = d["POSITION"] * d["RETURN"] - d["TRADES"] * cost
    d["CUM_STRATEGY"] = np.exp(d["STRATEGY"].cumsum())
    d["CUM_BH"] = np.exp(d["RETURN"].cumsum())

    _, dd = max_drawdown(d["CUM_STRATEGY"])
    perf = pd.DataFrame(
        [_perf_row(d["STRATEGY"], d["CUM_STRATEGY"]),
         _perf_row(d["RETURN"], d["CUM_BH"])],
        index=[f"Momentum {window}d", "Buy & Hold"],
    ).round(2)

    sig_chg = d["SIGNAL"].diff()
    return {
        "df": d, "dd": dd, "perf": perf, "window": window,
        "n_trades": int(d["TRADES"].sum()),
        "buys": d[(sig_chg > 0) & (d["SIGNAL"] == 1)],
        "sells": d[(sig_chg < 0) & (d["SIGNAL"] == -1)],
        "mc": montecarlo_var(d["STRATEGY"], d["CUM_STRATEGY"].iloc[-1]),
        "label": f"Momentum {window}d",
    }


# ════════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA 4 · REBALANCEO PERIÓDICO vs BUY & HOLD
# ════════════════════════════════════════════════════════════════════════════════
def backtest_rebalanceo(adj_close: pd.DataFrame, weights: np.ndarray,
                        freq: str = "ME", cost: float = 0.0015,
                        rf: float = RF_ANUAL_DEFAULT, is_frac: float = 0.8,
                        capital: float = 1.0) -> dict:
    tickers = list(adj_close.columns)
    returns = adj_close.pct_change().dropna()
    insample = int(len(returns) * is_frac)
    rt = returns[insample:].copy()
    rt.index = pd.to_datetime(rt.index)

    holdings = pd.DataFrame(index=rt.index, columns=tickers, dtype=float)
    holdings.loc[rt.index[0]] = weights * capital
    rebal_dates = rt.resample(freq).last().index

    port_hist, turnover_hist, cost_acum = [], [], 0.0
    for i, date in enumerate(rt.index[1:], 1):
        prev = rt.index[i - 1]
        holdings.loc[date] = holdings.loc[prev] * (1 + rt.loc[date])
        if date in rebal_dates:
            total = holdings.loc[date].sum()
            pesos_antes = holdings.loc[date] / total
            turnover = (pesos_antes - weights).abs().sum()
            costo = turnover * total * cost
            holdings.loc[date] = weights * (total - costo)
            cost_acum += costo
            turnover_hist.append({"date": date, "turnover": turnover, "cost": costo})
        port_hist.append(holdings.loc[date].sum())

    bh_values = (1 + rt).cumprod() * weights * capital
    bh_port = bh_values.sum(axis=1)

    res = pd.DataFrame({"Rebalanceo": port_hist,
                        "BuyHold": bh_port[1:].values}, index=rt.index[1:])
    res["ret_reb"] = res["Rebalanceo"].pct_change().fillna(0)
    res["ret_bh"] = res["BuyHold"].pct_change().fillna(0)

    _, dd_reb = max_drawdown(res["Rebalanceo"])
    _, dd_bh = max_drawdown(res["BuyHold"])
    perf = pd.DataFrame(
        [_perf_row(res["ret_reb"], res["Rebalanceo"] / capital, arit=True),
         _perf_row(res["ret_bh"], res["BuyHold"] / capital, arit=True)],
        index=[f"Rebalanceo {freq}", "Buy & Hold"],
    ).round(2)

    bh_pesos = bh_values.div(bh_values.sum(axis=1), axis=0)
    return {
        "df": res, "dd_reb": dd_reb, "dd_bh": dd_bh, "perf": perf,
        "freq": freq, "n_rebal": len(turnover_hist), "cost_acum": cost_acum,
        "capital": capital, "bh_pesos": bh_pesos, "rebal_dates": rebal_dates,
        "label": f"Rebalanceo {freq}",
    }


# ════════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA 5 · MACHINE LEARNING + WALK-FORWARD
# ════════════════════════════════════════════════════════════════════════════════
ML_MODELS = ["DecisionTree", "RandomForest", "XGBoost", "SVM"]


def _make_model(choice: str, final: bool = False):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    if choice == "DecisionTree":
        return DecisionTreeClassifier(max_depth=5, random_state=42)
    if choice == "RandomForest":
        n = 100 if final else 10
        return RandomForestClassifier(n_estimators=n, max_depth=3, random_state=42)
    if choice == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(eval_metric="logloss", max_depth=5, random_state=42)
    if choice == "SVM":
        return SVC(probability=True, random_state=42)
    raise ValueError(f"Modelo no válido: {choice}")


def backtest_ml(adj_close: pd.DataFrame, weights: np.ndarray,
                model_choice: str = "RandomForest", window: int = 252,
                cost: float = 0.0015, rf: float = RF_ANUAL_DEFAULT) -> dict:
    """
    Clasificación binaria (sube/no-sube mañana) con Walk-Forward Analysis.
    Features = media móvil 5d de retornos por activo (shift 1). Opera long
    cuando predice subida, cash si no. Devuelve métricas + matriz de confusión.
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

    returns = np.log(adj_close / adj_close.shift(1)).dropna()
    X = returns.rolling(window=5).mean().shift(1).dropna()
    y = (returns.sum(axis=1) > 0).astype(int).shift(-1).dropna()
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    y_pred, y_true, strat_rets = [], [], []
    for start in range(0, len(X) - window):
        end = start + window
        model = _make_model(model_choice)
        model.fit(X.iloc[start:end], y.iloc[start:end])
        pred = model.predict(X.iloc[end:end + 1])[0]
        y_pred.append(int(pred))
        y_true.append(int(y.iloc[end:end + 1].values[0]))
        daily = (returns.loc[X.index[end]] * weights).sum()
        strat_rets.append(daily if pred == 1 else 0.0)

    res = pd.DataFrame(index=X.index[window:])
    res["Signal"] = y_pred
    res["Truth"] = y_true
    res["Strategy_Gross"] = strat_rets
    res["BuyHold"] = (returns.loc[res.index] * weights).sum(axis=1)
    res["Trades"] = res["Signal"].diff().abs().fillna(0)
    res["Strategy"] = res["Strategy_Gross"] - res["Trades"] * cost
    res["CUM_ML"] = np.exp(res["Strategy"].cumsum())
    res["CUM_BH"] = np.exp(res["BuyHold"].cumsum())
    res.index = pd.to_datetime(res.index)

    _, dd = max_drawdown(res["CUM_ML"])
    perf = pd.DataFrame(
        [_perf_row(res["Strategy"], res["CUM_ML"]),
         _perf_row(res["BuyHold"], res["CUM_BH"])],
        index=[f"ML {model_choice}", "Buy & Hold"],
    ).round(2)

    cm = confusion_matrix(res["Truth"], res["Signal"])
    acc = accuracy_score(res["Truth"], res["Signal"])

    # Importancia de features (si el modelo la expone)
    importances = None
    if model_choice in ("DecisionTree", "RandomForest", "XGBoost"):
        fm = _make_model(model_choice, final=True)
        fm.fit(X, y)
        importances = pd.Series(fm.feature_importances_, index=X.columns).sort_values()

    monthly = res["Strategy"].resample("ME").sum()
    return {
        "df": res, "dd": dd, "perf": perf, "cm": cm, "acc": acc,
        "importances": importances, "monthly": monthly,
        "model": model_choice, "window": window,
        "n_trades": int(res["Trades"].sum()),
        "days_in": int((res["Signal"] == 1).sum()), "n_total": len(res),
        "mc": montecarlo_var(res["Strategy"], res["CUM_ML"].iloc[-1]),
        "report": classification_report(res["Truth"], res["Signal"],
                                        target_names=["Bajada", "Subida"],
                                        output_dict=True, zero_division=0),
        "label": f"ML {model_choice}",
    }
