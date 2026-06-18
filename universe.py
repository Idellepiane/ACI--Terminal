"""
universe.py — Catálogo de instrumentos financieros disponibles.

Combina dos fuentes:
  1. Listas curadas estáticas: cientos de tickers pre-cargados, agrupados
     por categoría (US equities, índices, ETFs, FX, commodities, crypto,
     ADRs argentinos, etc.).
  2. Búsqueda dinámica vía FMP /stable/search-symbol para encontrar
     cualquier ticker fuera de la lista (incluye prácticamente toda
     bolsa global cubierta por FMP, ~70k instrumentos).
"""

from __future__ import annotations

import requests
import pandas as pd

from fundamentals import FMP_API_KEY, FMP_BASE


# ════════════════════════════════════════════════════════════════════════════════
# UNIVERSOS CURADOS
# ════════════════════════════════════════════════════════════════════════════════

# US equities — Top 200 por capitalización + tech / financials / healthcare etc.
US_LARGECAPS = [
    # Tech mega
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "ORCL", "ADBE", "CRM", "AMD", "INTC", "QCOM", "TXN", "INTU", "CSCO",
    "IBM", "NOW", "PANW", "SNOW", "PLTR", "ANET", "MU", "AMAT", "LRCX",
    "KLAC", "ADI", "MRVL", "NXPI", "WDAY", "DDOG", "NET", "CRWD", "ZS",
    "MDB", "OKTA", "TEAM", "DOCU", "U", "RBLX", "DASH", "ABNB", "UBER",
    "LYFT", "SHOP", "SPOT", "SQ", "PYPL", "COIN", "HOOD", "SOFI",
    # Communication
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    # Consumer
    "WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX", "MCD", "CMG", "YUM",
    "KO", "PEP", "PG", "CL", "KMB", "MDLZ", "MO", "PM", "BUD", "DEO",
    "EL", "ULTA", "LULU", "RH", "BBY", "ROST", "TJX", "DG", "DLTR",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "BX", "KKR", "APO", "BRK-B", "V", "MA", "AIG",
    "MET", "PRU", "TRV", "ALL", "CB", "PGR", "SPGI", "MCO", "ICE", "CME",
    "NDAQ", "MSCI", "MKTX",
    # Healthcare / pharma / biotech
    "UNH", "JNJ", "LLY", "MRK", "PFE", "ABBV", "TMO", "ABT", "DHR", "ISRG",
    "BMY", "AMGN", "GILD", "CVS", "CI", "HUM", "ELV", "MCK", "BDX", "SYK",
    "MDT", "EW", "BSX", "ZBH", "RMD", "VRTX", "REGN", "BIIB", "ILMN",
    "MRNA", "BNTX", "NVAX",
    # Energy
    "XOM", "CVX", "COP", "OXY", "EOG", "SLB", "PSX", "MPC", "VLO", "PXD",
    "HES", "FANG", "DVN", "BKR", "HAL", "WMB", "KMI", "EPD", "ET", "MPLX",
    # Industrials
    "BA", "CAT", "DE", "HON", "GE", "RTX", "LMT", "NOC", "GD", "UPS",
    "FDX", "UNP", "CSX", "NSC", "DAL", "UAL", "AAL", "LUV", "MMM", "EMR",
    "ETN", "ITW", "PH", "ROK", "DOV", "FTV", "JCI", "OTIS", "URI",
    # Materials / chemicals
    "LIN", "APD", "ECL", "SHW", "DD", "DOW", "PPG", "EMN", "ALB", "FCX",
    "NEM", "NUE", "STLD", "CLF", "X", "VMC", "MLM",
    # Utilities / real estate
    "NEE", "SO", "DUK", "AEP", "EXC", "XEL", "SRE", "D", "AMT", "PLD",
    "EQIX", "CCI", "PSA", "O", "SPG", "WELL", "DLR", "AVB", "EQR",
    # Autos
    "F", "GM", "STLA", "TM", "RIVN", "LCID", "NIO",
]

# Major US indices (Yahoo notation)
US_INDICES = [
    "^GSPC",   # S&P 500
    "^DJI",    # Dow Jones
    "^IXIC",   # NASDAQ Composite
    "^NDX",    # NASDAQ 100
    "^RUT",    # Russell 2000
    "^VIX",    # VIX
    "^TNX",    # 10Y Treasury yield
    "^TYX",    # 30Y Treasury yield
    "^FVX",    # 5Y Treasury yield
    "^IRX",    # 13W Treasury bill
]

# Major global indices
WORLD_INDICES = [
    # Europa
    "^FTSE",      # FTSE 100 UK
    "^GDAXI",     # DAX 40 Germany
    "^FCHI",      # CAC 40 France
    "^STOXX50E",  # Euro Stoxx 50
    "^STOXX",     # Stoxx Europe 600
    "^IBEX",      # IBEX 35 Spain
    "FTSEMIB.MI", # FTSE MIB Italia
    "^AEX",       # AEX Holanda
    "^SSMI",      # SMI Suiza
    "^OMX",       # OMX S30 Suecia
    "^BFX",       # BEL 20 Bélgica
    # EMEA
    "XU100.IS",   # BIST 100 Turquía
    "^TA125.TA",  # TA-125 Israel
    "^TASI.SR",   # Tadawul Arabia Saudita
    # Asia-Pacífico
    "^N225",      # Nikkei 225
    "^HSI",       # Hang Seng
    "000001.SS",  # Shanghai Composite
    "399001.SZ",  # Shenzhen Composite
    "000300.SS",  # CSI 300 China
    "^KS11",      # KOSPI Corea
    "^TWII",      # TAIEX Taiwán
    "^BSESN",     # BSE Sensex India
    "^NSEI",      # NIFTY 50 India
    "^AXJO",      # S&P/ASX 200 Australia
    "^NZ50",      # NZX 50 Nueva Zelanda
    "^STI",       # Straits Times Singapur
    "^JKSE",      # IDX Composite Indonesia
    "^KLSE",      # FTSE KLCI Malasia
    "PSEI.PS",    # PSEi Filipinas
    # Américas
    "^GSPTSE",    # S&P/TSX Canadá
    "^BVSP",      # Bovespa Brasil
    "^MXX",       # IPC México
    "^MERV",      # Merval Argentina
    "^IPSA",      # S&P IPSA Chile
]

# ETFs populares (broad market, sectores, factor, internacional, bonos)
ETFS = [
    # Broad
    "SPY", "IVV", "VOO", "QQQ", "DIA", "IWM", "VTI", "ITOT", "RSP",
    # Sectors (SPDR)
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
    # International
    "EFA", "VEA", "IEFA", "VWO", "EEM", "IEMG", "FXI", "MCHI", "EWZ", "EWJ", "INDA", "ASHR",
    # Bonds
    "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "TIP", "MUB", "EMB",
    # Commodities / themes
    "GLD", "IAU", "SLV", "USO", "UNG", "DBC", "PDBC",
    # Factor / smart beta
    "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "VIG", "VYM", "SCHD", "DGRO",
    # Thematic
    "ARKK", "ARKW", "ARKG", "ARKQ", "ARKX", "ICLN", "TAN", "LIT", "URA", "SOXX", "SMH",
    "IBB", "XBI", "KBE", "KRE", "ITA", "JETS",
    # Leveraged / inverse
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "TMF", "TMV",
]

# Forex (Yahoo notation: PAIR=X)
FOREX = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
    "EURGBP=X", "EURJPY=X", "EURCHF=X", "GBPJPY=X", "AUDJPY=X",
    "USDARS=X", "USDBRL=X", "USDMXN=X", "USDCLP=X", "USDCOP=X", "USDPEN=X",
    "USDCNY=X", "USDINR=X", "USDKRW=X", "USDHKD=X", "USDSGD=X", "USDTRY=X", "USDZAR=X",
    "DX-Y.NYB",   # DXY Dollar Index
]

# Commodities (futures continuous)
COMMODITIES = [
    "CL=F",   # Crude Oil WTI
    "BZ=F",   # Brent
    "NG=F",   # Nat Gas
    "HO=F",   # Heating Oil
    "RB=F",   # Gasoline
    "GC=F",   # Gold
    "SI=F",   # Silver
    "PL=F",   # Platinum
    "PA=F",   # Palladium
    "HG=F",   # Copper
    "ZC=F",   # Corn
    "ZS=F",   # Soybean
    "ZW=F",   # Wheat
    "KC=F",   # Coffee
    "SB=F",   # Sugar
    "CC=F",   # Cocoa
    "CT=F",   # Cotton
    "LE=F",   # Live Cattle
    "HE=F",   # Lean Hogs
]

# Crypto (Yahoo notation: TICKER-USD)
CRYPTO = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD", "TRX-USD",
    "LTC-USD", "BCH-USD", "ATOM-USD", "UNI-USD", "ETC-USD", "XLM-USD",
    "NEAR-USD", "ICP-USD", "FIL-USD", "ARB-USD", "OP-USD", "INJ-USD",
    "SHIB-USD", "PEPE-USD", "WIF-USD",
]

# Empresas argentinas — ADRs en NYSE
ADR_ARG = [
    "YPF", "GGAL", "BMA", "PAM", "EDN", "CRESY", "TEO", "TGS",
    "LOMA", "IRS", "BBAR", "SUPV", "CEPU", "CAAP",
    "MELI",  # Mercado Libre (técnicamente uruguaya pero op. en LATAM)
    "GLOB",  # Globant
    "DESP",  # Despegar
    "VIST",  # Vista Energy
]

# Acciones argentinas (BYMA, sufijo .BA)
BA_STOCKS = [
    "GGAL.BA", "YPFD.BA", "PAMP.BA", "BMA.BA", "BBAR.BA", "EDN.BA",
    "CRES.BA", "TECO2.BA", "TGSU2.BA", "LOMA.BA", "IRSA.BA", "SUPV.BA",
    "CEPU.BA", "TRAN.BA", "ALUA.BA", "TXAR.BA", "MIRG.BA", "VALO.BA",
    "COME.BA", "BYMA.BA", "CVH.BA", "BHIP.BA", "AGRO.BA", "HARG.BA",
    "MOLI.BA", "AUSO.BA", "LEDE.BA", "GCLA.BA", "GCDI.BA", "GBAN.BA",
    "ROSE.BA", "MORI.BA",
    # ETFs / índices Argentina
    "M.BA",  # Merval
]

# CEDEARs populares en BYMA
CEDEARS = [
    "AAPL.BA", "MSFT.BA", "AMZN.BA", "GOOGL.BA", "META.BA", "TSLA.BA",
    "NVDA.BA", "NFLX.BA", "DIS.BA", "KO.BA", "MCD.BA", "JNJ.BA",
    "PFE.BA", "JPM.BA", "BAC.BA", "WMT.BA", "PG.BA", "V.BA", "MA.BA",
    "BA.BA", "XOM.BA", "CVX.BA",
]


# Categorías agrupadas para la UI
UNIVERSE: dict[str, list[str]] = {
    "US LARGE CAPS": US_LARGECAPS,
    "US INDICES": US_INDICES,
    "WORLD INDICES": WORLD_INDICES,
    "ETFs": ETFS,
    "FX": FOREX,
    "COMMODITIES": COMMODITIES,
    "CRYPTO": CRYPTO,
    "ARG ADRs (NYSE)": ADR_ARG,
    "ARG EQUITIES (BYMA)": BA_STOCKS,
    "CEDEARS": CEDEARS,
}


# Lista plana, deduplicada, ordenada — alimenta el multiselect principal
def all_tickers() -> list[str]:
    seen: set[str] = set()
    flat: list[str] = []
    for group in UNIVERSE.values():
        for t in group:
            if t not in seen:
                seen.add(t)
                flat.append(t)
    return flat


# ════════════════════════════════════════════════════════════════════════════════
# BÚSQUEDA DINÁMICA  ·  FMP /stable/search-symbol
# ════════════════════════════════════════════════════════════════════════════════
def search_fmp(query: str, limit: int = 25) -> pd.DataFrame:
    """
    Devuelve un DataFrame (symbol, name, currency, exchange) de instrumentos
    que matchean el query en FMP. Cubre ~70k tickers globales.
    """
    if not query or len(query.strip()) < 1:
        return pd.DataFrame()

    url = f"{FMP_BASE}/search-symbol"
    params = {"query": query.strip(), "limit": limit, "apikey": FMP_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return pd.DataFrame()

    if not data or not isinstance(data, list):
        return pd.DataFrame()

    df = pd.DataFrame(data)
    keep = [c for c in ["symbol", "name", "currency", "exchangeShortName", "exchange"] if c in df.columns]
    return df[keep].rename(columns={"exchangeShortName": "exchange_short"})
