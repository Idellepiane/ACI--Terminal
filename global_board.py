"""
global_board.py — Tablero global estilo Bloomberg (WEI / GMM).

Pantalla de mercados mundiales: índices bursátiles por región, FX (majors,
LATAM y EM), commodities (energía, metales, agro), tasas US, volatilidad,
crypto y un panel Argentina con CCL implícito vía ADRs.

Todo se renderiza como HTML denso (tablas con sparklines SVG inline) sobre
el CSS Bloomberg de app.py, más un par de figuras Plotly (curva de tasas y
treemap mundial). Este módulo no pega contra APIs: recibe el snapshot de
cotizaciones (qmap) y las series para sparklines ya descargadas por app.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go


# ════════════════════════════════════════════════════════════════════════════════
# CATÁLOGO DEL TABLERO  ·  (símbolo Yahoo, nombre display, sub-etiqueta)
# ════════════════════════════════════════════════════════════════════════════════

EQUITY_REGIONS: dict[str, list[tuple[str, str, str]]] = {
    "AMÉRICAS": [
        ("^GSPC", "S&P 500", "USA"),
        ("^DJI", "DOW JONES", "USA"),
        ("^IXIC", "NASDAQ COMP", "USA"),
        ("^NDX", "NASDAQ 100", "USA"),
        ("^RUT", "RUSSELL 2000", "USA"),
        ("^GSPTSE", "S&P/TSX", "CANADÁ"),
        ("^BVSP", "BOVESPA", "BRASIL"),
        ("^MXX", "S&P/BMV IPC", "MÉXICO"),
        ("^MERV", "MERVAL", "ARGENTINA"),
        ("^IPSA", "S&P IPSA", "CHILE"),
    ],
    "EUROPA · EMEA": [
        ("^STOXX50E", "EURO STOXX 50", "EUROZONA"),
        ("^STOXX", "STOXX 600", "EUROPA"),
        ("^FTSE", "FTSE 100", "REINO UNIDO"),
        ("^GDAXI", "DAX 40", "ALEMANIA"),
        ("^FCHI", "CAC 40", "FRANCIA"),
        ("^IBEX", "IBEX 35", "ESPAÑA"),
        ("FTSEMIB.MI", "FTSE MIB", "ITALIA"),
        ("^AEX", "AEX 25", "HOLANDA"),
        ("^SSMI", "SMI 20", "SUIZA"),
        ("^OMX", "OMX S30", "SUECIA"),
        ("^BFX", "BEL 20", "BÉLGICA"),
        ("XU100.IS", "BIST 100", "TURQUÍA"),
        ("^TA125.TA", "TA-125", "ISRAEL"),
        ("^TASI.SR", "TADAWUL", "ARABIA S."),
    ],
    "ASIA · PACÍFICO": [
        ("^N225", "NIKKEI 225", "JAPÓN"),
        ("^HSI", "HANG SENG", "HONG KONG"),
        ("000001.SS", "SHANGHAI COMP", "CHINA"),
        ("399001.SZ", "SHENZHEN COMP", "CHINA"),
        ("000300.SS", "CSI 300", "CHINA"),
        ("^KS11", "KOSPI", "COREA SUR"),
        ("^TWII", "TAIEX", "TAIWÁN"),
        ("^NSEI", "NIFTY 50", "INDIA"),
        ("^BSESN", "SENSEX", "INDIA"),
        ("^AXJO", "S&P/ASX 200", "AUSTRALIA"),
        ("^NZ50", "NZX 50", "N. ZELANDA"),
        ("^STI", "STRAITS TIMES", "SINGAPUR"),
        ("^JKSE", "IDX COMP", "INDONESIA"),
        ("^KLSE", "FTSE KLCI", "MALASIA"),
        ("PSEI.PS", "PSEi", "FILIPINAS"),
    ],
}

FX_PANELS: dict[str, list[tuple[str, str, str]]] = {
    "FX · MAJORS": [
        ("DX-Y.NYB", "DXY INDEX", "USD"),
        ("EURUSD=X", "EUR/USD", "EURO"),
        ("GBPUSD=X", "GBP/USD", "LIBRA"),
        ("USDJPY=X", "USD/JPY", "YEN"),
        ("USDCHF=X", "USD/CHF", "FRANCO"),
        ("USDCAD=X", "USD/CAD", "CAD"),
        ("AUDUSD=X", "AUD/USD", "AUD"),
        ("NZDUSD=X", "NZD/USD", "NZD"),
        ("EURGBP=X", "EUR/GBP", "CROSS"),
        ("EURJPY=X", "EUR/JPY", "CROSS"),
        ("GBPJPY=X", "GBP/JPY", "CROSS"),
    ],
    "FX · LATAM & EM": [
        ("USDARS=X", "USD/ARS", "ARGENTINA"),
        ("USDBRL=X", "USD/BRL", "BRASIL"),
        ("USDMXN=X", "USD/MXN", "MÉXICO"),
        ("USDCLP=X", "USD/CLP", "CHILE"),
        ("USDCOP=X", "USD/COP", "COLOMBIA"),
        ("USDPEN=X", "USD/PEN", "PERÚ"),
        ("USDUYU=X", "USD/UYU", "URUGUAY"),
        ("USDCNY=X", "USD/CNY", "CHINA"),
        ("USDINR=X", "USD/INR", "INDIA"),
        ("USDKRW=X", "USD/KRW", "COREA"),
        ("USDTRY=X", "USD/TRY", "TURQUÍA"),
        ("USDZAR=X", "USD/ZAR", "SUDÁFRICA"),
    ],
    "CRYPTO · 24/7": [
        ("BTC-USD", "BITCOIN", "BTC"),
        ("ETH-USD", "ETHEREUM", "ETH"),
        ("SOL-USD", "SOLANA", "SOL"),
        ("BNB-USD", "BNB", "BNB"),
        ("XRP-USD", "XRP", "XRP"),
        ("ADA-USD", "CARDANO", "ADA"),
        ("DOGE-USD", "DOGECOIN", "DOGE"),
        ("AVAX-USD", "AVALANCHE", "AVAX"),
    ],
}

COMMODITY_PANELS: dict[str, list[tuple[str, str, str]]] = {
    "ENERGÍA": [
        ("CL=F", "WTI CRUDE", "NYMEX"),
        ("BZ=F", "BRENT", "ICE"),
        ("NG=F", "NAT GAS", "NYMEX"),
        ("RB=F", "GASOLINA RBOB", "NYMEX"),
        ("HO=F", "HEATING OIL", "NYMEX"),
    ],
    "METALES": [
        ("GC=F", "ORO", "COMEX"),
        ("SI=F", "PLATA", "COMEX"),
        ("HG=F", "COBRE", "COMEX"),
        ("PL=F", "PLATINO", "NYMEX"),
        ("PA=F", "PALADIO", "NYMEX"),
        ("ALI=F", "ALUMINIO", "COMEX"),
    ],
    "AGRO": [
        ("ZS=F", "SOJA", "CBOT"),
        ("ZC=F", "MAÍZ", "CBOT"),
        ("ZW=F", "TRIGO", "CBOT"),
        ("ZL=F", "ACEITE SOJA", "CBOT"),
        ("ZM=F", "HARINA SOJA", "CBOT"),
        ("KC=F", "CAFÉ", "ICE"),
        ("SB=F", "AZÚCAR", "ICE"),
        ("CC=F", "CACAO", "ICE"),
        ("CT=F", "ALGODÓN", "ICE"),
        ("LE=F", "GANADO VIVO", "CME"),
    ],
}

RATES = [
    ("^IRX", "T-BILL 13W", 0.25),
    ("^FVX", "UST 5Y", 5.0),
    ("^TNX", "UST 10Y", 10.0),
    ("^TYX", "UST 30Y", 30.0),
]

VOL_PANEL: list[tuple[str, str, str]] = [
    ("^VIX", "VIX", "S&P 500"),
    ("^VXN", "VXN", "NASDAQ 100"),
    ("^VVIX", "VVIX", "VOL DEL VIX"),
    ("^MOVE", "MOVE", "BONOS US"),
]

# ADRs argentinos para el panel local. El ratio (acciones BYMA por ADR) se usa
# para el CCL implícito; sólo GGAL e YPF se usan para el cálculo mostrado.
ARG_ADRS: list[tuple[str, str, str]] = [
    ("GGAL", "GRUPO GALICIA", "BANCOS"),
    ("BMA", "BANCO MACRO", "BANCOS"),
    ("BBAR", "BBVA ARG", "BANCOS"),
    ("SUPV", "SUPERVIELLE", "BANCOS"),
    ("YPF", "YPF", "ENERGÍA"),
    ("PAM", "PAMPA ENERGÍA", "ENERGÍA"),
    ("CEPU", "CENTRAL PUERTO", "ENERGÍA"),
    ("EDN", "EDENOR", "ENERGÍA"),
    ("TGS", "TRANSP. GAS SUR", "ENERGÍA"),
    ("VIST", "VISTA ENERGY", "ENERGÍA"),
    ("LOMA", "LOMA NEGRA", "MATERIALES"),
    ("CRESY", "CRESUD", "AGRO"),
    ("IRS", "IRSA", "REAL ESTATE"),
    ("TEO", "TELECOM ARG", "TELCO"),
    ("MELI", "MERCADO LIBRE", "TECH"),
    ("GLOB", "GLOBANT", "TECH"),
]

# Pares (local BYMA, ADR, ratio) para CCL implícito
CCL_PAIRS = [
    ("GGAL.BA", "GGAL", 10.0),
    ("YPFD.BA", "YPF", 1.0),
]

# Cinta animada superior (subset de lo más mirado)
TAPE = [
    "^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^TNX", "DX-Y.NYB",
    "EURUSD=X", "USDJPY=X", "GC=F", "CL=F", "BZ=F", "HG=F", "ZS=F",
    "BTC-USD", "ETH-USD", "^STOXX50E", "^FTSE", "^GDAXI", "^N225",
    "^HSI", "^BVSP", "^MERV", "USDARS=X", "USDBRL=X",
]

# Horarios cash de cada plaza (hora local, lun-vie; pausas de almuerzo ignoradas)
SESSIONS = [
    ("SÍDNEY", "Australia/Sydney", (10, 0), (16, 0), "ASX"),
    ("TOKIO", "Asia/Tokyo", (9, 0), (15, 0), "TSE"),
    ("SHANGHÁI", "Asia/Shanghai", (9, 30), (15, 0), "SSE"),
    ("HONG KONG", "Asia/Hong_Kong", (9, 30), (16, 0), "HKEX"),
    ("MUMBAI", "Asia/Kolkata", (9, 15), (15, 30), "NSE"),
    ("FRÁNCFORT", "Europe/Berlin", (9, 0), (17, 30), "XETRA"),
    ("LONDRES", "Europe/London", (8, 0), (16, 30), "LSE"),
    ("NUEVA YORK", "America/New_York", (9, 30), (16, 0), "NYSE"),
    ("BUENOS AIRES", "America/Argentina/Buenos_Aires", (11, 0), (17, 0), "BYMA"),
]


def _flat_board() -> list[tuple[str, str, str, str]]:
    """(symbol, nombre, sub, grupo) de todos los paneles del tablero."""
    out: list[tuple[str, str, str, str]] = []
    for region, rows in EQUITY_REGIONS.items():
        out += [(s, n, sub, region) for s, n, sub in rows]
    for grp, rows in {**FX_PANELS, **COMMODITY_PANELS}.items():
        out += [(s, n, sub, grp) for s, n, sub in rows]
    out += [(s, n, sub, "VOLATILIDAD") for s, n, sub in VOL_PANEL]
    out += [(s, n, "TASAS", "TASAS US") for s, n, _ in RATES]
    out += [(s, n, sub, "ARG ADRs") for s, n, sub in ARG_ADRS]
    return out


BOARD_FLAT = _flat_board()

# Tickers extra que necesita el board pero no se muestran como fila propia
_EXTRA = [loc for loc, _, _ in CCL_PAIRS]

ALL_BOARD_TICKERS: tuple[str, ...] = tuple(dict.fromkeys(
    [s for s, _, _, _ in BOARD_FLAT] + _EXTRA
))


# ════════════════════════════════════════════════════════════════════════════════
# CSS ADICIONAL  ·  cinta animada, chips de sesiones, tablas densas
# ════════════════════════════════════════════════════════════════════════════════
GLOBAL_CSS = """
<style>
    @keyframes gb-tape-scroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .gb-tape {
        overflow: hidden; white-space: nowrap;
        background: #050505;
        border: 1px solid #2a2a2a; border-left: 4px solid #d97a00;
        padding: 5px 0; margin-bottom: 8px;
    }
    .gb-tape-inner { display: inline-block; animation: gb-tape-scroll 90s linear infinite; }
    .gb-tape:hover .gb-tape-inner { animation-play-state: paused; }
    .gb-tape-item { padding: 0 16px; font-size: 12px; }

    .gb-sess { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .gb-chip {
        background: #0a0a0a; border: 1px solid #2a2a2a;
        padding: 4px 10px; font-size: 10px; letter-spacing: 0.5px;
        display: flex; align-items: center;
    }
    .gb-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; }
    .gb-dot-open { background: #00ff41; box-shadow: 0 0 6px #00ff41; }
    .gb-dot-closed { background: #ff2050; }
    .gb-chip .city { color: #d97a00; font-weight: 700; margin-right: 8px; }
    .gb-chip .hms { color: #fff; font-weight: 700; margin-right: 8px; }
    .gb-chip .st-open { color: #00ff41; font-weight: 700; }
    .gb-chip .st-closed { color: #666; }

    .gb-table { width: 100%; border-collapse: collapse; font-size: 11px; }
    .gb-table td { padding: 3px 6px; border-bottom: 1px solid #161616; vertical-align: middle; }
    .gb-table tr:hover td { background: #131313; }
    .gb-name { color: #e8e8e8; font-weight: 700; white-space: nowrap; }
    .gb-sub { color: #5f5f5f; font-size: 9px; letter-spacing: 0.5px; }
    .gb-px { color: #fff; font-weight: 700; text-align: right; white-space: nowrap; }
    .gb-chg { text-align: right; white-space: nowrap; font-size: 11px; }
    .gb-spark { text-align: right; width: 70px; }

    .gb-ars { display: flex; gap: 6px; margin-bottom: 6px; }
    .gb-ars-cell {
        flex: 1; background: #0a0a0a; border: 1px solid #2a2a2a;
        border-left: 3px solid #d97a00; padding: 5px 8px;
    }
    .gb-ars-cell .l { color: #d97a00; font-size: 9px; letter-spacing: 1px; font-weight: 700; }
    .gb-ars-cell .v { color: #fff; font-size: 15px; font-weight: 700; }
    .gb-ars-cell .s { font-size: 9px; }

    .gb-mover-bar { height: 10px; display: inline-block; }

    /* --- Flash de tick (precio sube/baja), estilo Bloomberg --- */
    .gb-table td, .gb-tape-item, .gb-ars-cell { transition: background-color 0.6s ease-out; }
    .gb-flash-up { background-color: rgba(0, 255, 65, 0.22) !important; transition: none !important; }
    .gb-flash-dn { background-color: rgba(255, 32, 80, 0.25) !important; transition: none !important; }

    /* --- Densidad: menos aire entre bloques (tablero aglomerado) --- */
    div[data-testid="stVerticalBlock"] { gap: 0.45rem !important; }
    div[data-testid="stElementContainer"] { margin-bottom: 0 !important; }
    .bbg-panel { margin-bottom: 4px; }
    .main .block-container { padding-top: 0.4rem; }
</style>
"""


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS DE FORMATO
# ════════════════════════════════════════════════════════════════════════════════
def _isnum(v) -> bool:
    try:
        return v is not None and not pd.isna(v)
    except (TypeError, ValueError):
        return False


def _fmt_px(v) -> str:
    if not _isnum(v):
        return "—"
    av = abs(v)
    if av >= 100_000:
        return f"{v:,.0f}"
    if av >= 1_000:
        return f"{v:,.2f}"
    if av >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _chg_span(pct) -> str:
    if not _isnum(pct):
        return "<span class='flat'>—</span>"
    if pct > 0:
        return f"<span class='up'>▲ +{pct:.2f}%</span>"
    if pct < 0:
        return f"<span class='down'>▼ {pct:.2f}%</span>"
    return "<span class='flat'>■ 0.00%</span>"


def quotes_map(quotes_df: pd.DataFrame) -> dict[str, dict]:
    """DataFrame de fetch_quotes → dict por ticker con last/chg/pct."""
    qmap: dict[str, dict] = {}
    if quotes_df is None or quotes_df.empty:
        return qmap
    for _, r in quotes_df.iterrows():
        if not _isnum(r.get("Último")):
            continue
        qmap[r["Ticker"]] = {
            "last": float(r["Último"]),
            "chg": float(r["Cambio"]) if _isnum(r.get("Cambio")) else None,
            "pct": float(r["Cambio %"]) if _isnum(r.get("Cambio %")) else None,
            "name": r.get("Nombre", r["Ticker"]),
        }
    return qmap


def spark_svg(series: pd.Series | None, w: int = 64, h: int = 18) -> str:
    """Sparkline SVG inline (verde si sube en el período, rojo si baja)."""
    if series is None or len(series) < 2:
        return ""
    y = series.astype(float).values
    lo, hi = float(y.min()), float(y.max())
    rng = (hi - lo) or 1.0
    n = len(y)
    pts = []
    for i, v in enumerate(y):
        px = 1 + i * (w - 2) / (n - 1)
        py = 2 + (h - 4) * (1 - (v - lo) / rng)
        pts.append(f"{px:.1f},{py:.1f}")
    color = "#00ff41" if y[-1] >= y[0] else "#ff2050"
    return (
        f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg'>"
        f"<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-width='1.2'/>"
        f"<circle cx='{pts[-1].split(',')[0]}' cy='{pts[-1].split(',')[1]}' r='1.6' fill='{color}'/>"
        f"</svg>"
    )


# ════════════════════════════════════════════════════════════════════════════════
# BLOQUES HTML
# ════════════════════════════════════════════════════════════════════════════════
def tape_html(qmap: dict[str, dict]) -> str:
    """Cinta animada superior con lo más mirado del mundo."""
    items = []
    for sym in TAPE:
        q = qmap.get(sym)
        if not q:
            continue
        label = _tape_label(sym)
        items.append(
            f"<span class='gb-tape-item'><b style='color:#d97a00'>{label}</b> "
            f"<span class='value gb-last' data-sym='{sym}'>{_fmt_px(q['last'])}</span> "
            f"<span class='gb-pct' data-sym='{sym}'>{_chg_span(q['pct'])}</span></span>"
        )
    inner = "".join(items)
    if not inner:
        return ""
    return f"<div class='gb-tape'><div class='gb-tape-inner'>{inner}{inner}</div></div>"


_TAPE_LABELS = {s: n for s, n, _, _ in BOARD_FLAT}


def _tape_label(sym: str) -> str:
    return _TAPE_LABELS.get(sym, sym)


def sessions_html(now_utc: datetime | None = None) -> str:
    """Chips de plazas del mundo con hora local y estado ABIERTO/CERRADO."""
    base = now_utc or datetime.now(ZoneInfo("UTC"))
    chips = []
    for city, tz, (oh, om), (ch, cm), venue in SESSIONS:
        loc = base.astimezone(ZoneInfo(tz))
        is_weekday = loc.weekday() < 5
        mins = loc.hour * 60 + loc.minute
        is_open = is_weekday and (oh * 60 + om) <= mins < (ch * 60 + cm)
        dot = "gb-dot-open" if is_open else "gb-dot-closed"
        status = "<span class='st-open'>ABIERTO</span>" if is_open else "<span class='st-closed'>CERRADO</span>"
        chips.append(
            f"<div class='gb-chip'><span class='gb-dot {dot}'></span>"
            f"<span class='city'>{city} · {venue}</span>"
            f"<span class='hms' data-tz='{tz}'>{loc.strftime('%H:%M')}</span>{status}</div>"
        )
    return f"<div class='gb-sess'>{''.join(chips)}</div>"


def panel_html(title: str, rows: list[tuple[str, str, str]],
               qmap: dict[str, dict], sparks: dict[str, pd.Series] | None = None) -> str:
    """Panel Bloomberg: tabla densa nombre / último / %chg / sparkline."""
    sparks = sparks or {}
    body_rows = []
    for sym, name, sub in rows:
        # Renderizamos SIEMPRE la fila (con ancla data-sym) aunque todavía no
        # haya dato: así el price stream la rellena cuando el poller la trae,
        # sin necesidad de re-render de toda la app. Placeholder = "···".
        q = qmap.get(sym)
        px = f"<span class='gb-last' data-sym='{sym}'>{_fmt_px(q['last']) if q else '···'}</span>"
        pct = f"<span class='gb-pct' data-sym='{sym}'>{_chg_span(q['pct']) if q else ''}</span>"
        body_rows.append(
            "<tr>"
            f"<td><span class='gb-name'>{name}</span><br><span class='gb-sub'>{sub} · {sym}</span></td>"
            f"<td class='gb-px'>{px}</td>"
            f"<td class='gb-chg'>{pct}</td>"
            f"<td class='gb-spark'>{spark_svg(sparks.get(sym))}</td>"
            "</tr>"
        )
    return (
        f"<div class='bbg-panel'><div class='bbg-panel-header'>{title}</div>"
        f"<table class='gb-table'>{''.join(body_rows)}</table></div>"
    )


def ars_summary_html(qmap: dict[str, dict]) -> str:
    """Mini-grid Argentina: USD oficial, CCL implícito (GGAL / YPF) y brecha."""
    oficial = qmap.get("USDARS=X", {}).get("last")
    ccls: dict[str, float] = {}
    for local, adr, ratio in CCL_PAIRS:
        pl, pa = qmap.get(local, {}).get("last"), qmap.get(adr, {}).get("last")
        if _isnum(pl) and _isnum(pa) and pa:
            ccls[adr] = pl * ratio / pa

    def _cell(key: str, label: str, value, sub: str = "") -> str:
        v = _fmt_px(value) if _isnum(value) else "—"
        return (f"<div class='gb-ars-cell' data-ars='{key}'><div class='l'>{label}</div>"
                f"<div class='v'>{v}</div><div class='s gb-sub'>{sub}</div></div>")

    ccl_ref = ccls.get("GGAL") or ccls.get("YPF")
    brecha = (ccl_ref / oficial - 1) * 100 if (_isnum(oficial) and oficial and _isnum(ccl_ref)) else None
    brecha_html = (
        f"<div class='gb-ars-cell' data-ars='brecha'><div class='l'>BRECHA</div>"
        f"<div class='v'>{_chg_span(brecha) if _isnum(brecha) else '—'}</div>"
        f"<div class='s gb-sub'>CCL vs OFICIAL</div></div>"
    )
    return (
        "<div class='gb-ars'>"
        + _cell("oficial", "USD OFICIAL", oficial, "USDARS=X · YAHOO")
        + _cell("ccl_ggal", "CCL IMPLÍCITO GGAL", ccls.get("GGAL"), "GGAL.BA ×10 / GGAL")
        + _cell("ccl_ypf", "CCL IMPLÍCITO YPF", ccls.get("YPF"), "YPFD.BA / YPF")
        + brecha_html
        + "</div>"
    )


def movers_html(qmap: dict[str, dict], top_n: int = 10) -> tuple[str, str]:
    """(panel ganadores, panel perdedores) sobre TODO el tablero global."""
    rows = []
    for sym, name, sub, grp in BOARD_FLAT:
        q = qmap.get(sym)
        if q and _isnum(q.get("pct")):
            rows.append((sym, name, grp, q["pct"], q["last"]))
    rows.sort(key=lambda r: r[3], reverse=True)
    gainers, losers = rows[:top_n], rows[-top_n:][::-1]
    max_abs = max((abs(r[3]) for r in gainers + losers), default=1) or 1

    def _tbl(data, color):
        trs = []
        for sym, name, grp, pct, last in data:
            bar_w = max(3, int(60 * abs(pct) / max_abs))
            trs.append(
                "<tr>"
                f"<td><span class='gb-name'>{name}</span><br><span class='gb-sub'>{grp} · {sym}</span></td>"
                f"<td class='gb-px'><span class='gb-last' data-sym='{sym}'>{_fmt_px(last)}</span></td>"
                f"<td class='gb-chg'><span class='gb-pct' data-sym='{sym}'>{_chg_span(pct)}</span></td>"
                f"<td class='gb-spark'><span class='gb-mover-bar' style='width:{bar_w}px;background:{color}'></span></td>"
                "</tr>"
            )
        return f"<table class='gb-table'>{''.join(trs)}</table>"

    g = f"<div class='bbg-panel'><div class='bbg-panel-header'>TOP GANADORES · HOY</div>{_tbl(gainers, '#00ff41')}</div>"
    l = f"<div class='bbg-panel'><div class='bbg-panel-header'>TOP PERDEDORES · HOY</div>{_tbl(losers, '#ff2050')}</div>"
    return g, l


# ════════════════════════════════════════════════════════════════════════════════
# PRICE STREAM  ·  actualización in-place del DOM (sin rerun de la página)
# ════════════════════════════════════════════════════════════════════════════════
# Se inyecta vía st.components.v1.html (iframe same-origin) desde un fragment
# minúsculo que corre cada ~2s. El script pisa los <span data-sym> del documento
# padre con los precios nuevos y hace flash verde/rojo en cada tick, estilo
# Bloomberg. Nada del resto de la página se re-renderiza: charts, scroll y
# dropdowns quedan intactos.
_STREAM_TEMPLATE = """
<script>
(function () {
    var data = __PAYLOAD__;
    var doc = window.parent.document;
    var prev = window.parent.__gbPrev = window.parent.__gbPrev || {};

    function fpx(v) {
        var av = Math.abs(v);
        if (av >= 100000) return v.toLocaleString('en-US', {maximumFractionDigits: 0});
        if (av >= 1000) return v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        if (av >= 10) return v.toFixed(2);
        return v.toFixed(4);
    }
    function pctHtml(p) {
        if (p === null || p === undefined || isNaN(p)) return "<span class='flat'>—</span>";
        if (p > 0) return "<span class='up'>▲ +" + p.toFixed(2) + "%</span>";
        if (p < 0) return "<span class='down'>▼ " + p.toFixed(2) + "%</span>";
        return "<span class='flat'>■ 0.00%</span>";
    }
    function flash(el, up) {
        var cell = el.closest('td') || el.closest('.gb-tape-item') || el;
        cell.classList.remove('gb-flash-up', 'gb-flash-dn');
        void cell.offsetWidth;  // reflow para reiniciar la transición
        cell.classList.add(up ? 'gb-flash-up' : 'gb-flash-dn');
        setTimeout(function () { cell.classList.remove('gb-flash-up', 'gb-flash-dn'); }, 500);
    }

    doc.querySelectorAll('.gb-last[data-sym]').forEach(function (el) {
        var d = data[el.dataset.sym];
        if (!d || d[0] === null || d[0] === undefined) return;
        var txt = fpx(d[0]);
        if (el.textContent !== txt) {
            var p = prev[el.dataset.sym];
            flash(el, p === undefined ? true : d[0] >= p);
            el.textContent = txt;
        }
    });
    doc.querySelectorAll('.gb-pct[data-sym]').forEach(function (el) {
        var d = data[el.dataset.sym];
        if (d) el.innerHTML = pctHtml(d[1]);
    });

    // ── Panel ARS: CCL implícito recalculado en el cliente ──────────────
    function last(s) { return data[s] ? data[s][0] : null; }
    var ofi = last('USDARS=X');
    var cclG = (last('GGAL.BA') && last('GGAL')) ? last('GGAL.BA') * 10 / last('GGAL') : null;
    var cclY = (last('YPFD.BA') && last('YPF')) ? last('YPFD.BA') / last('YPF') : null;
    function setArs(k, v) {
        var el = doc.querySelector("[data-ars='" + k + "'] .v");
        if (el && v !== null) el.textContent = fpx(v);
    }
    setArs('oficial', ofi); setArs('ccl_ggal', cclG); setArs('ccl_ypf', cclY);
    var ref = cclG || cclY;
    if (ofi && ref) {
        var el = doc.querySelector("[data-ars='brecha'] .v");
        if (el) el.innerHTML = pctHtml((ref / ofi - 1) * 100);
    }

    Object.keys(data).forEach(function (s) { if (data[s][0] !== null) prev[s] = data[s][0]; });

    // ── Relojes vivos: header (con segundos) y plazas del mundo ─────────
    if (window.parent.__gbClock) clearInterval(window.parent.__gbClock);
    window.parent.__gbClock = setInterval(function () {
        var clk = doc.querySelector('.bbg-clock');
        if (clk) clk.textContent = new Date().toLocaleTimeString('en-GB');
        doc.querySelectorAll('.gb-chip .hms[data-tz]').forEach(function (el) {
            try {
                el.textContent = new Date().toLocaleTimeString('en-GB',
                    {timeZone: el.dataset.tz, hour: '2-digit', minute: '2-digit'});
            } catch (e) {}
        });
    }, 1000);
})();
</script>
<!-- tick __SALT__ -->
"""


def stream_js(payload: dict[str, list], salt: str) -> str:
    """
    HTML/JS para el fragment de streaming. payload = {sym: [last, pct]}.
    salt fuerza re-ejecución del iframe en cada tick.
    """
    return (_STREAM_TEMPLATE
            .replace("__PAYLOAD__", json.dumps(payload))
            .replace("__SALT__", salt))


# ════════════════════════════════════════════════════════════════════════════════
# FIGURAS PLOTLY
# ════════════════════════════════════════════════════════════════════════════════
def curve_fig(qmap: dict[str, dict], base_layout: dict) -> go.Figure | None:
    """Curva de Treasuries US (hoy vs cierre anterior) desde ^IRX/^FVX/^TNX/^TYX."""
    xs, ys, prevs, labels = [], [], [], []
    for sym, label, tenor in RATES:
        q = qmap.get(sym)
        if not q or not _isnum(q["last"]):
            continue
        xs.append(tenor)
        ys.append(q["last"])
        prevs.append(q["last"] - q["chg"] if _isnum(q.get("chg")) else None)
        labels.append(label)
    if len(xs) < 2:
        return None
    fig = go.Figure()
    if all(p is not None for p in prevs):
        fig.add_trace(go.Scatter(
            x=xs, y=prevs, mode="lines+markers", name="CIERRE ANT.",
            line=dict(color="#666", width=1.2, dash="dash"), marker=dict(size=5),
            hovertemplate="%{y:.3f}%<extra>cierre ant.</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers+text", name="HOY",
        line=dict(color="#d97a00", width=2.2), marker=dict(size=7, color="#d97a00"),
        text=[f"{v:.2f}" for v in ys], textposition="top center",
        textfont=dict(size=10, color="#fff"),
        hovertemplate="%{y:.3f}%<extra>hoy</extra>",
    ))
    layout = dict(base_layout)
    layout["height"] = 260
    fig.update_layout(**layout)
    fig.update_xaxes(title="Tenor (años)", type="log",
                     tickvals=xs, ticktext=[l.replace("UST ", "").replace("T-BILL ", "") for l in labels])
    fig.update_yaxes(title="Yield %")
    return fig


def treemap_fig(qmap: dict[str, dict], font_family: str) -> go.Figure | None:
    """Treemap del mundo: regiones/asset classes coloreados por % del día."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for region, rows in EQUITY_REGIONS.items():
        groups[region] = [(s, n) for s, n, _ in rows]
    groups["FX"] = [(s, n) for s, n, _ in FX_PANELS["FX · MAJORS"] + FX_PANELS["FX · LATAM & EM"]]
    groups["CRYPTO"] = [(s, n) for s, n, _ in FX_PANELS["CRYPTO · 24/7"]]
    groups["COMMODITIES"] = [(s, n) for rows in COMMODITY_PANELS.values() for s, n, _ in rows]

    ids, labels, parents, values, colors, texts = ["GLOBAL"], ["MERCADOS GLOBALES"], [""], [0], [0.0], [""]
    for grp, members in groups.items():
        present = [(s, n) for s, n in members if s in qmap and _isnum(qmap[s].get("pct"))]
        if not present:
            continue
        ids.append(grp); labels.append(grp); parents.append("GLOBAL")
        values.append(0); colors.append(0.0); texts.append("")
        for sym, name in present:
            pct = qmap[sym]["pct"]
            ids.append(sym); labels.append(name); parents.append(grp)
            values.append(1); colors.append(max(-3.0, min(3.0, pct)))
            texts.append(f"{pct:+.2f}%")
    if len(ids) <= 1:
        return None

    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values,
        text=texts, textinfo="label+text",
        textfont=dict(family=font_family, size=12, color="#fff"),
        marker=dict(
            colors=colors,
            colorscale=[[0.0, "#ff2050"], [0.5, "#141414"], [1.0, "#00ff41"]],
            cmin=-3, cmax=3, line=dict(color="#000", width=1),
        ),
        hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
        pathbar=dict(visible=True, textfont=dict(color="#d97a00")),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000",
        font=dict(family=font_family, color="#c8c8c8", size=11),
        margin=dict(l=4, r=4, t=28, b=4), height=460,
    )
    return fig
