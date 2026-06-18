# Bloomberg-style Dashboard · ACI I

Tablero de control financiero que integra todos los scripts de **Clase 1** (Portfolio + Estados Contables) y **Clase 2** (14 indicadores técnicos) en una sola app interactiva, estilo terminal Bloomberg, con datos en tiempo (cuasi) real.

## Qué tiene

- **🌐 GLOBAL** — Tablero mundial estilo Bloomberg WEI/GMM, independiente de la watchlist: ~117 instrumentos siempre visibles.
  - Índices bursátiles de ~30 países agrupados por región (Américas / Europa-EMEA / Asia-Pacífico), con sparklines del último mes.
  - FX majors, LATAM & EM, crypto; commodities (energía, metales, agro); curva de Treasuries US (hoy vs cierre anterior); panel de volatilidad (VIX, VXN, VVIX, MOVE).
  - Panel Argentina: ADRs en NYSE + USD oficial, **CCL implícito** (GGAL.BA×10/GGAL y YPFD.BA/YPF) y brecha.
  - Cinta animada de cotizaciones, relojes de las 9 plazas principales con estado ABIERTO/CERRADO, top movers globales y treemap mundial coloreado por % del día.
- **🌍 Mercado** — Cotizaciones live, variación %, volumen y mini-gráficos por ticker.
- **📈 Análisis Técnico** — 14 indicadores: Bollinger, Keltner, RSI, MACD, CCI, Estocástico, MFI, OBV, Aroon, OAD (Williams), Momentum, Hurst, Bandas de Auto-Regresión y Ondas de Elliott.
- **📊 Fundamentales** — Ratios TTM (P/E, P/B, EV/EBITDA, P/CF) y estados contables (Balance, Income, Cash Flow) vía Financial Modeling Prep.
- **💼 Portfolio** — Pesos, retornos acumulados, drawdown y correlaciones.
- **⚠️ RISK** (Clase 4) — VaR/CVaR por 3 métodos (paramétrico, histórico, Monte Carlo multivariado) con comparación y distribución; Performance Attribution Brinson (BHB vs BF) con tabla de sectores editable.
- **🧪 STRATEGY LAB** (Clase 5) — Backtesting con IS/OOS, costos y métricas (CAGR/Sharpe/Sortino/MaxDD/Calmar/Hit): Trend (cruce de medias con optimización + overfitting gap), Mean Reversion (RSI), Momentum, Rebalanceo periódico vs Buy&Hold, y Machine Learning Walk-Forward (Árbol / Random Forest / XGBoost / SVM) con matriz de confusión e importancia de features.
- **⏱ Live stream** — Siempre activo, sin configuración: los precios laten cada 5 segundos (flash verde/rojo por tick) sin recargar la página; los relojes tickean cada segundo. Usa el endpoint batched `spark` de Yahoo (2 requests por tick para todo el tablero) con backoff automático ante rate-limits.

## Instalación

```bash
cd "Dashboard"
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

## Configuración

La API key de FMP ya viene cargada de tus scripts originales (en `fundamentals.py`). Si querés cambiarla, editá `FMP_API_KEY` ahí o creá un `.env` con `FMP_API_KEY=tu_clave`.

## Cómo correrlo

```bash
streamlit run app.py
```

Se abre solo en `http://localhost:8501`.

## Notas

- Datos de precios: `yahooquery` (Yahoo Finance). El "tiempo real" es cuasi-real (delay típico de Yahoo de ~15 min en US equities). Para tick real necesitarías un feed pago (IEX Cloud, Polygon, Refinitiv).
- Datos fundamentales: Financial Modeling Prep API (`/stable/`).
- El auto-refresh refresca toda la página; los componentes pesados están cacheados con `@st.cache_data` para no re-pegarle a las APIs en cada tick.
