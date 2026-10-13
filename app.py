import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analizador de CEDEARs", layout="wide")

st.title("📊 Analizador & Escáner de CEDEARs con Filtros Avanzados")

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
    return yf.download(tickers_clean, period="1y", group_by="ticker", threads=True, progress=False)

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

            if len(df) < 150:
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

            # Medias Móviles
            sma21 = close.rolling(21).mean().iloc[-1]
            sma30 = close.rolling(30).mean().iloc[-1]
            ema150 = close.ewm(span=150, adjust=False).mean().iloc[-1]

            dist_ma21 = ((precio - sma21) / sma21) * 100
            dist_ma30 = ((precio - sma30) / sma30) * 100
            dist_ema150 = ((precio - ema150) / ema150) * 100

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
                "Dist. EMA150 %": round(dist_ema150, 2),
                "Oportunidad": oportunidad
            })
        except Exception:
            continue
    return pd.DataFrame(filas)

# --- 3. FUNCIÓN DE FILTRADO MÚLTIPLE ---
def aplicar_filtros_multiples(df, key_prefix):
    with st.expander("🔍 **Configurar Filtros Múltiples (Buscar entre valores)**"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            busqueda_ticker = st.text_input("Filtrar por Ticker:", key=f"{key_prefix}_ticker").upper().strip()
            min_rsi, max_rsi = st.slider("RSI (14) entre:", 0.0, 100.0, (0.0, 100.0), key=f"{key_prefix}_rsi")
            
        with c2:
            val_min_vol = float(df["Volatilidad (20d) %"].min()) if not df.empty else 0.0
            val_max_vol = float(df["Volatilidad (20d) %"].max()) if not df.empty else 100.0
            min_vol, max_vol = st.slider("Volatilidad (20d) % entre:", val_min_vol, val_max_vol, (val_min_vol, val_max_vol), key=f"{key_prefix}_vol")

            val_min_var = float(df["Var. Diaria %"].min()) if not df.empty else -20.0
            val_max_var = float(df["Var. Diaria %"].max()) if not df.empty else 20.0
            min_var, max_var = st.slider("Var. Diaria % entre:", val_min_var, val_max_var, (val_min_var, val_max_var), key=f"{key_prefix}_var")

        with c3:
            val_min_e150 = float(df["Dist. EMA150 %"].min()) if not df.empty else -50.0
            val_max_e150 = float(df["Dist. EMA150 %"].max()) if not df.empty else 50.0
            min_ema150, max_ema150 = st.slider("Dist. EMA150 % entre:", val_min_e150, val_max_e150, (val_min_e150, val_max_e150), key=f"{key_prefix}_e150")

    df_filtrado = df.copy()

    if busqueda_ticker:
        df_filtrado = df_filtrado[df_filtrado["Ticker"].str.contains(busqueda_ticker)]

    df_filtrado = df_filtrado[
        (df_filtrado["RSI (14)"] >= min_rsi) & (df_filtrado["RSI (14)"] <= max_rsi) &
        (df_filtrado["Volatilidad (20d) %"] >= min_vol) & (df_filtrado["Volatilidad (20d) %"] <= max_vol) &
        (df_filtrado["Var. Diaria %"] >= min_var) & (df_filtrado["Var. Diaria %"] <= max_var) &
        (df_filtrado["Dist. EMA150 %"] >= min_ema150) & (df_filtrado["Dist. EMA150 %"] <= max_ema150)
    ]

    return df_filtrado

# --- 4. PROCESAMIENTO GENERAL ---
todos_tickers = list(set(BASE_VOLATILES + st.session_state["watchlist"]))

with st.spinner("Conectando con Yahoo Finance y calculando indicadores..."):
    datos_raw = descargar_datos(todos_tickers)
    df_master = procesar_indicadores(datos_raw, todos_tickers)

# --- 5. INTERFAZ EN PESTAÑAS ---
tab_volatiles, tab_watchlist = st.tabs(["🔥 Top 100 Más Volátiles", "📌 Mi Watchlist Personalizada"])

with tab_volatiles:
    st.subheader("Top 100 CEDEARs con Mayor Volatilidad del Mercado")
    if not df_master.empty:
        df_vol = df_master.sort_values(by="Volatilidad (20d) %", ascending=False).head(100)
        
        # Aplicar panel de filtros
        df_vol_filtrado = aplicar_filtros_multiples(df_vol, key_prefix="top100")
        
        st.caption(f"Showing **{len(df_vol_filtrado)}** de {len(df_vol)} CEDEARs. Encabezados fijos al hacer scroll.")
        st.dataframe(df_vol_filtrado, use_container_width=True, hide_index=True, height=600)

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
        # Aplicar panel de filtros
        df_watch_filtrado = aplicar_filtros_multiples(df_watch, key_prefix="watchlist")
        
        st.caption(f"Showing **{len(df_watch_filtrado)}** de {len(df_watch)} CEDEARs en Watchlist.")
        st.dataframe(df_watch_filtrado, use_container_width=True, hide_index=True, height=600)
    else:
        st.info("No hay tickers en tu watchlist.")
