# app.py — Unified Dashboard (FINAL FIX - IMPORT STATS ADDED)
import streamlit as st
import pandas as pd
import numpy as np
import os
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # <--- INI YANG TADI KURANG
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Dashboard Saham BUMN", page_icon="📈", layout="wide")
DATA_FOLDER = "data_saham_bumn"
os.makedirs(DATA_FOLDER, exist_ok=True)

TICKERS = ['BBRI.JK', 'BBNI.JK', 'BMRI.JK', 'BBTN.JK']
BANK_CODES = [t.replace(".JK", "") for t in TICKERS]
CSV_FILES = {bank: os.path.join(DATA_FOLDER, f"{bank}_2014_2024.csv") for bank in BANK_CODES}
MASTER_CSV = os.path.join(DATA_FOLDER, "combined_2014_2024.csv")

BANK_COLORS = {
    'BBRI': '#00529C', # BRI Biru
    'BBNI': '#00A1E0', # BNI Tosca
    'BMRI': '#F3A800', # Mandiri Emas
    'BBTN': '#EC1C24'  # BTN Merah
}

# -------------------------
# HELPERS
# -------------------------
def detect_price_col(df: pd.DataFrame) -> Optional[str]:
    cols = [str(c).lower().replace(" ", "") for c in df.columns]
    mapping = dict(zip(cols, df.columns))
    for name in ("adjclose", "adj_close", "close"):
        if name in mapping:
            return mapping[name]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return None

# -------------------------
# SCRAPING & SAVE
# -------------------------
def scrape_and_save(tickers: List[str] = TICKERS, start: str = "2014-01-01", end: str = "2024-12-31") -> bool:
    st.info("🔄 Mengunduh data dari Yahoo Finance...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False)
    
    if raw is None or raw.empty:
        st.error("Gagal mendownload data.")
        return False
        
    raw = raw.reset_index()
    saved = []
    
    for ticker in tickers:
        bank = ticker.replace(".JK", "")
        try:
            if isinstance(raw.columns, pd.MultiIndex) or any(isinstance(c, tuple) for c in raw.columns):
                df = pd.DataFrame({
                    "Date": raw["Date"],
                    "Open": raw["Open"][ticker],
                    "High": raw["High"][ticker],
                    "Low": raw["Low"][ticker],
                    "Close": raw["Close"][ticker],
                    "Adj Close": raw["Adj Close"][ticker],
                    "Volume": raw["Volume"][ticker],
                })
            else:
                tmp = raw.copy()
                cols_map = {c: c for c in tmp.columns} 
                df = tmp 
            
            cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for c in cols_to_numeric:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            df = df.dropna().set_index("Date")
            df.to_csv(CSV_FILES[bank])
            saved.append(bank)
        except Exception as e:
            st.warning(f"Gagal memproses/menyimpan {bank}: {e}")
    
    all_list = []
    for b in saved:
        d = safe_read_csv(CSV_FILES[b])
        if d is not None:
            d2 = d.reset_index()
            d2['Bank'] = b
            all_list.append(d2)
            
    if all_list:
        pd.concat(all_list, ignore_index=True).to_csv(MASTER_CSV, index=False)
        st.success(f"Selesai mengunduh & menyimpan: {', '.join(saved)}")
        return True
    return False

# -------------------------
# LOAD CSV (cached)
# -------------------------
@st.cache_data
def load_all_csvs() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    datasets = {}
    all_rows = []
    for bank, path in CSV_FILES.items():
        if os.path.exists(path):
            df = safe_read_csv(path)
            if df is not None:
                if not isinstance(df.index, pd.DatetimeIndex):
                     df.index = pd.to_datetime(df.index)
                
                datasets[bank] = df
                tmp = df.copy().reset_index()
                tmp['Bank'] = bank
                all_rows.append(tmp)
                
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
    else:
        combined = pd.DataFrame()
        
    return datasets, combined

# -------------------------
# METRIC CALCULATION
# -------------------------
@st.cache_data
def get_all_metrics(_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics_list = []
    for bank, dfb in _datasets.items():
        if dfb is None or dfb.empty: continue
        pc = detect_price_col(dfb)
        if pc is None: continue
            
        df = dfb.copy()
        df[pc] = pd.to_numeric(df[pc], errors='coerce')
        df = df.dropna(subset=[pc])
        
        df['Return'] = df[pc].pct_change()
        df['Volatility_30d'] = df['Return'].rolling(window=30).std() * np.sqrt(252)
        df['Cumulative_Return'] = (1 + df['Return']).cumprod()
        df['Daily_Return_Pct'] = df['Return'] * 100
        df['Harga'] = df[pc]
        
        df = df.dropna(subset=['Return', 'Volatility_30d', 'Cumulative_Return', 'Daily_Return_Pct'])
        df = df.reset_index()
        df['Bank'] = bank
        metrics_list.append(df)
    
    if not metrics_list: return pd.DataFrame()
    return pd.concat(metrics_list, ignore_index=True)

def outliers_zscore(series: pd.Series, thresh: float = 3.0) -> pd.Series:
    s = series.dropna().astype(float)
    if len(s) < 2: return pd.Series(dtype=float)
    z = np.abs(stats.zscore(s)) # Ini sekarang akan berhasil karena stats sudah diimpor
    return s[z > thresh]

# -------------------------
# UI: Sidebar
# -------------------------
st.sidebar.title("Kontrol Data")
if st.sidebar.button("Update Data (Scrape Ulang dari Yahoo)"):
    ok = scrape_and_save()
    if ok:
        load_all_csvs.clear()
        get_all_metrics.clear()
        st.rerun()

st.sidebar.write("Data folder:", DATA_FOLDER)
st.sidebar.caption("Mode: Hybrid (Load CSV, scrape jika perlu)")

# -------------------------
# MAIN APP: DATA LOADING
# -------------------------
datasets = {}
combined_raw = pd.DataFrame()

datasets, combined_raw = load_all_csvs()
if combined_raw.empty:
    st.info("CSV tidak ditemukan — akan otomatis mengunduh data pertama kali.")
    ok = scrape_and_save()
    if not ok:
        st.error("Scraping gagal; tidak ada data.")
        st.stop()
    load_all_csvs.clear()
    datasets, combined_raw = load_all_csvs()

if combined_raw is None or combined_raw.empty:
    st.error("Tidak ada data. Lakukan Update Data.")
    st.stop()
    
df_metrics = get_all_metrics(datasets)

# -------------------------
# LAYOUT
# -------------------------
st.title("📈 Dashboard Analisis Saham Bank BUMN (2014-2024)")
st.markdown("Dibuat oleh **KARASUNO**")
st.markdown("---")

filter_col1, filter_col2 = st.columns(2) 
with filter_col1:
    bank_options = sorted(combined_raw['Bank'].unique())
    selected_banks = st.multiselect("Pilih bank", bank_options, default=bank_options)

with filter_col2:
    combined_raw['Date'] = pd.to_datetime(combined_raw['Date'])
    date_min_limit = combined_raw['Date'].min().date()
    date_max_limit = combined_raw['Date'].max().date()
    
    date_range = st.date_input("Pilih rentang tanggal", [date_min_limit, date_max_limit], min_value=date_min_limit, max_value=date_max_limit)

if len(date_range) != 2:
    st.warning("Pilih rentang tanggal valid.")
    st.stop()

date_mask_raw = (combined_raw['Date'].dt.date >= date_range[0]) & (combined_raw['Date'].dt.date <= date_range[1])
bank_mask_raw = combined_raw['Bank'].isin(selected_banks)
df_filtered_raw = combined_raw[bank_mask_raw & date_mask_raw].copy()

df_metrics['Date'] = pd.to_datetime(df_metrics['Date'])
date_mask_metrics = (df_metrics['Date'].dt.date >= date_range[0]) & (df_metrics['Date'].dt.date <= date_range[1])
bank_mask_metrics = df_metrics['Bank'].isin(selected_banks)
df_filtered_metrics = df_metrics[bank_mask_metrics & date_mask_metrics].copy()

st.markdown("---")
st.metric(label="📄 Total Baris Data (Hasil Filter)", value=f"{len(df_filtered_raw):,}")
st.markdown("---")

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data & EDA", "📈 Charts", "📦 Hist", "🚨 Extreme & Outlier", "💡 Recommendation"])

with tab1:
    st.subheader("Sample Data Mentah")
    st.dataframe(df_filtered_raw.head(200))
    st.dataframe(df_filtered_raw.select_dtypes(include=[np.number]).describe().T)

with tab2:
    st.subheader("Tren Harga")
    price_col = detect_price_col(df_filtered_raw)
    fig_price = px.line(df_filtered_raw, x='Date', y=price_col, color='Bank', title=f"Tren Harga ({price_col})", color_discrete_map=BANK_COLORS)
    fig_price.update_layout(hovermode="x unified", template="plotly_white", height=520, xaxis_title="Tahun", yaxis_title="Harga (IDR)")
    fig_price.update_yaxes(tickprefix="Rp ")
    fig_price.update_xaxes(dtick="M12", tickformat="%Y")
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")
    st.subheader("Analisis Lanjutan")
    if not df_filtered_metrics.empty:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=("Harga", "Volatilitas 30D", "Cumulative Return"))
        for bank in selected_banks:
            d = df_filtered_metrics[df_filtered_metrics['Bank']==bank]
            if d.empty: continue
            col = BANK_COLORS.get(bank, None)
            fig.add_trace(go.Scatter(x=d['Date'], y=d['Harga'], mode='lines', name=bank, line=dict(color=col), hovertemplate="%{x|%Y-%m-%d}<br>Rp %{y:,.0f}"), row=1, col=1)
            fig.add_trace(go.Scatter(x=d['Date'], y=d['Volatility_30d'], mode='lines', name=bank, showlegend=False, line=dict(color=col)), row=2, col=1)
            fig.add_trace(go.Scatter(x=d['Date'], y=d['Cumulative_Return'], mode='lines', name=bank, showlegend=False, line=dict(color=col)), row=3, col=1)
        
        fig.update_layout(height=900, hovermode="x unified", template="plotly_white", xaxis3_showticklabels=True, xaxis3_dtick="M12", xaxis3_tickformat="%Y", xaxis3_title="Tahun")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Histogram")
    col = detect_price_col(df_filtered_raw)
    fig = px.histogram(df_filtered_raw, x=col, color='Bank', barmode='overlay', nbins=60, color_discrete_map=BANK_COLORS)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Boxplot")
    df_melt = df_filtered_raw.melt(id_vars=['Bank'], value_vars=['Open','High','Low','Close','Adj Close','Volume'], var_name='Fitur', value_name='Nilai')
    fig_box = px.box(df_melt, x='Bank', y='Nilai', color='Bank', facet_col='Fitur', color_discrete_map=BANK_COLORS)
    fig_box.update_yaxes(matches=None, type='log')
    st.plotly_chart(fig_box, use_container_width=True)

with tab4:
    st.subheader("Outlier Detection (Data FULL 2014-2024)")
    zt = 3
    st.info(f"Analisis outlier menggunakan Z-score > {zt} pada **SELURUH DATA HISTORIS** (mengabaikan filter tanggal di atas) untuk menjaga konsistensi fakta sejarah.")

    outlier_summary = []
    
    for bank in selected_banks: 
        if bank not in datasets: continue
            
        df_full_raw = datasets[bank].copy() 
        pc = detect_price_col(df_full_raw)
        if pc is None: continue
        
        df_full_raw[pc] = pd.to_numeric(df_full_raw[pc], errors='coerce')
        df_clean = df_full_raw.dropna(subset=[pc])
        
        outs = outliers_zscore(df_clean[pc], thresh=zt)
        
        outlier_summary.append({
            "Bank": bank, "PriceCol": pc, "Outliers": len(outs), "Total Data": len(df_clean)
        })
        
        st.markdown(f"### {bank}")
        start_d = df_clean.index.min().strftime('%Y-%m-%d')
        end_d = df_clean.index.max().strftime('%Y-%m-%d')
        st.caption(f"Menganalisis data dari {start_d} sampai {end_d}")
        
        st.write(f"Ditemukan **{len(outs)}** outlier.")
        
        if len(outs) > 0:
            with st.expander("Lihat Detail Data"):
                st.dataframe(outs.to_frame(name="Harga").head(22))
            
            fig_out = go.Figure()
            col = BANK_COLORS.get(bank, 'blue')
            fig_out.add_trace(go.Scatter(x=df_clean.index, y=df_clean[pc], mode='lines', name='Harga', line=dict(color=col, width=1)))
            fig_out.add_trace(go.Scatter(x=outs.index, y=outs.values, mode='markers', marker=dict(color='red', size=6, symbol='x'), name='Outlier'))
            fig_out.update_layout(height=400, title=f"Posisi Outlier {bank} (Full History)", template="plotly_white")
            st.plotly_chart(fig_out, use_container_width=True)
        else:
            st.success("Tidak ada outlier ekstrem.")

    st.markdown("---")
    st.subheader("Extreme Movements (Daily Return)")
    extremes = []
    for bank in selected_banks:
        if bank not in datasets: continue
        d_raw = datasets[bank].copy()
        pc = detect_price_col(d_raw)
        d_raw[pc] = pd.to_numeric(d_raw[pc], errors='coerce')
        d_raw['Ret'] = d_raw[pc].pct_change() * 100
        d_raw = d_raw.dropna(subset=['Ret'])
        
        idxmax = d_raw['Ret'].idxmax()
        idxmin = d_raw['Ret'].idxmin()
        
        extremes.append({
            "Bank": bank,
            "Max Spike": f"{d_raw.loc[idxmax, 'Ret']:.2f}% ({idxmax.date()})",
            "Max Drop": f"{d_raw.loc[idxmin, 'Ret']:.2f}% ({idxmin.date()})"
        })
    st.dataframe(pd.DataFrame(extremes))

with tab5:
    st.subheader("Recommendation")
    stats = []
    for bank in selected_banks:
        dfb = df_filtered_metrics[df_filtered_metrics['Bank'] == bank]
        if dfb.empty: continue
        start_p = dfb.iloc[0]['Harga']
        end_p = dfb.iloc[-1]['Harga']
        ret = (end_p / start_p) - 1
        vol = dfb['Return'].std() * np.sqrt(252)
        stats.append({'Bank': bank, 'TotalReturn': ret, 'AnnualVol': vol})
        
    if stats:
        sdf = pd.DataFrame(stats).sort_values('TotalReturn', ascending=False)
        sdf['TotalReturn%'] = (sdf['TotalReturn']*100).round(2).astype(str) + '%'
        sdf['AnnualVol%'] = (sdf['AnnualVol']*100).round(2).astype(str) + '%'
        st.dataframe(sdf)
        st.markdown(f"**Top Pick:** {sdf.iloc[0]['Bank']}")

st.markdown("---")
st.caption("Dashboard app: Final Version (Fixed Import & Outliers).")