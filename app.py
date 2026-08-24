import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analizador de CEDEARs", layout="wide")

st.title("📊 Analizador & Escáner de CEDEARs (Tiempo Real)")

# --- 1. LISTA DE TICKERS ---
WATCHLIST_INICIAL = [
    "AVGO", "MSTR", "COIN", "NOW", "TQQQ", "HUT", "BITO", "ANF", 
    "GOOGL", "NVDA", "C", "MSFT", "KRE", "HMY", "SLV", "WDC", 
    "BABA", "AMD", "HL", "GLW"
]

BASE_VOLATILES = sorted(list(set(WATCHLIST_INICIAL + [
    "RIOT", "MARA", "PLTR", "TSLA", "MELI", "PATH", "RBLX", "U", "DOCU", "ZM", 
    "ROKU", "SNAP", "PINS", "UBER", "ABNB", "DASH", "SE", "SHOP", "SQ", "PYPL", 
    "NFLX", "SPOT", "CRWD", "ZS", "NET", "DDOG", "SNOW", "MDB", "PLUG", "FCEL", 
    "ENPH", "SEDG", "FSLR", "PANW", "BNTX", "MRNA", "XBI", "ARKG", "GILD", 
    "AMGN", "PFE", "BIIB", "NIO", "XPEV", "PDD", "JD", "BIDU", "NTES", "TCEHY", 
    "EEM", "EWZ", "AAL", "UAL", "DAL", "CCL", "RCL", "NCLH", "DIS", "BKNG", 
    "MAR", "SBUX", "NKE", "XLE", "USO", "UNG", "CLF", "FCX", "X", "AA", "PBR", 
    "VIST", "TEN", "TX", "VALE", "RIO", "BHP", "XOM", "CVX", "SHEL", "NEM", 
    "ARKK", "SPY", "QQQ", "IWM", "DIA", "XLF", "AAPL", "AMZN", "META", "CRM"
])))

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = WATCHLIST_INICIAL

# --- 2. DESCARGA Y CÁLCULOS TÉCNICOS ---
@st.cache_data(ttl=600)
def descargar_datos(tickers):
    tickers_clean = sorted(list(set([t.upper().strip() for t in tickers if t])))
    return yf.download(tickers_clean, period="6mo", group_by="ticker", threads=True, progress=False)

def procesar_indicadores(datos, tickers):
    filas = []
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                if ticker not in datos.columns.levels[0]:
                    continue
                df = datos[ticker].dropna()
            else:
                df = datos.dropna()

            if len(df) < 35:
                continue

            close = df['Close']
            volume = df['Volume']
            precio = close.iloc[-1]
            precio_prev = close.iloc[-2]
            var_pct = ((precio - precio_prev) / precio_prev) * 100

            # RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1]))

            # MACD (12, 26, 9)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            hist = macd_line - signal_line

            if hist.iloc[-2] < 0 and hist.iloc[-1] > 0:
                macd_txt = "🟢 Cruce Alcista"
            elif hist.iloc[-1] > 0:
                macd_txt = "🟢 Alcista"
            elif hist.iloc[-2] > 0 and hist.iloc[-1] < 0:
                macd_txt = "🔴 Cruce Bajista"
            else:
                macd_txt = "🔴 Bajista"

            # Medias Móviles 21 y 30
            sma21 = close.rolling(21).mean().iloc[-1]
            sma30 = close.rolling(30).mean().iloc[-1]
            dist_ma21 = ((precio - sma21) / sma21) * 100
            dist_ma30 = ((precio - sma30) / sma30) * 100

            # Tendencia
            if precio > sma21 and sma21 > sma30:
                tendencia = "🟢 Alcista Fuerte"
            elif precio > sma21:
                tendencia = "🟢 Alcista"
            elif precio < sma21 and sma21 < sma30:
                tendencia = "🔴 Bajista Fuerte"
            else:
                tendencia = "🟡 Lateral"

            # Volatilidad Anualizada 20d
            log_ret = np.log(close / close.shift(1))
            volatibilidad = log_ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100

            # Volumen Relativo y Manos Grandes
            vol_sma20 = volume.rolling(20).mean().iloc[-1]
            vol_rel = volume.iloc[-1] / vol_sma20 if vol_sma20 > 0 else 1.0

            if vol_rel >= 1.5 and var_pct > 0:
                manos_grandes = "🐋 Acumulación"
            elif vol_rel >= 1.5 and var_pct < 0:
                manos_grandes = "🔴 Distribución"
            elif vol_rel >= 1.1:
                manos_grandes = "🟢 Compra Leve"
            else:
                manos_grandes = "⚪ Neutro"

            # Oportunidad Compra / Venta
            if rsi <= 35 and dist_ma21 < -3 and "Alcista" in macd_txt:
                oportunidad = "🟢 Compra Clara"
            elif rsi <= 40 or "Acumulación" in manos_grandes:
                oportunidad = "🟢 Oportunidad"
            elif rsi >= 70 or dist_ma21 > 8:
                oportunidad = "🔴 Sobrecompra / Venta"
            else:
                oportunidad = "⚪ Neutral"

            filas.append({
                "Ticker": ticker,
                "Precio USD": round(precio, 2),
                "Var. Diaria %": round(var_pct, 2),
                "Volatilidad (20d) %": round(volatibilidad, 1),
                "MACD": macd_txt,
                "RSI (14)": round(rsi, 1),
                "Tendencia": tendencia,
                "Volumen Rel.": round(vol_rel, 2),
                "Manos Grandes": manos_grandes,
                "Dist. MA21 %": round(dist_ma21, 2),
                "Dist. MA30 %": round(dist_ma30, 2),
                "Oportunidad": oportunidad
            })
        except Exception:
            continue
    return pd.DataFrame(filas)

# --- 3. PROCESAMIENTO GENERAL ---
todos_tickers = list(set(BASE_VOLATILES + st.session_state["watchlist"]))

with st.spinner("Conectando con Yahoo Finance y calculando indicadores..."):
    datos_raw = descargar_datos(todos_tickers)
    df_master = procesar_indicadores(datos_raw, todos_tickers)

# --- 4. INTERFAZ EN PESTAÑAS ---
tab_volatiles, tab_watchlist = st.tabs(["🔥 Top 100 Más Volátiles", "📌 Mi Watchlist Personalizada"])

with tab_volatiles:
    st.subheader("Top 100 CEDEARs con Mayor Volatilidad del Mercado")
    if not df_master.empty:
        df_vol = df_master.sort_values(by="Volatilidad (20d) %", ascending=False).head(100)
        st.caption("💡 *Hacé clic sobre la cabecera de cualquier columna para ordenar los datos.*")
        st.dataframe(df_vol, use_container_width=True, hide_index=True)

with tab_watchlist:
    st.subheader("Mi Lista de Seguimiento Personalizada")
    
    col_in, col_bt = st.columns([3, 1])
    with col_in:
        nuevo = st.text_input("Agregar Ticker a la Watchlist:", "").upper().strip()
    with col_bt:
        st.write("")
        st.write("")
        if st.button("➕ Agregar Ticker", use_container_width=True):
            if nuevo and nuevo not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(nuevo)
                st.rerun()

    df_watch = df_master[df_master["Ticker"].isin(st.session_state["watchlist"])].copy()
    if not df_watch.empty:
        st.caption("💡 *Hacé clic sobre la cabecera de cualquier columna para ordenar los datos.*")
        st.dataframe(df_watch, use_container_width=True, hide_index=True)
    else:
        st.info("No hay tickers en tu watchlist.")