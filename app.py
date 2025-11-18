# app3.py — Clean Dashboard (Professional UI + Terminology Fix: "Outlier")
import streamlit as st
import pandas as pd
import numpy as np
import os
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime
from typing import Dict, List, Optional

# -------------------------
# 1. CONFIG & DESIGN SYSTEM
# -------------------------
st.set_page_config(page_title="Executive Dashboard BUMN", page_icon="📊", layout="wide")

# Warna Branding (Patterns)
BANK_COLORS = {
    'BBRI': '#00529C', # Biru BRI
    'BBNI': '#005E6A', # Tosca BNI 
    'BMRI': '#F3A800', # Emas Mandiri
    'BBTN': '#D6001C'  # Merah BTN
}

DATA_FOLDER = "data_saham_bumn"
os.makedirs(DATA_FOLDER, exist_ok=True)
TICKERS = ['BBRI.JK', 'BBNI.JK', 'BMRI.JK', 'BBTN.JK']
CSV_FILES = {t.replace(".JK", ""): os.path.join(DATA_FOLDER, f"{t.replace('.JK', '')}_2014_2024.csv") for t in TICKERS}
MASTER_CSV = os.path.join(DATA_FOLDER, "combined_2014_2024.csv")

# -------------------------
# 2. VISUALIZATION HELPER
# -------------------------
def style_chart(fig, title="", x_title="", y_title=""):
    """Menerapkan prinsip Data-Ink Ratio: Hapus grid berlebih, perjelas label."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template="plotly_white", # White space yang bersih
        hovermode="x unified",   # Memudahkan perbandingan (Alignment)
        height=450,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="") # Legenda rapi di atas
    )
    # Hapus grid vertikal (Noise), pertahankan horizontal untuk panduan nilai
    fig.update_xaxes(showgrid=False, title=x_title, tickformat="%Y") 
    fig.update_yaxes(showgrid=True, gridcolor='#eee', title=y_title)
    return fig

# -------------------------
# 3. DATA PROCESSING
# -------------------------
def detect_price_col(df):
    for c in df.columns:
        if str(c).lower().replace(" ", "") in ["adjclose", "adj_close", "close"]:
            return c
    return df.select_dtypes(include=[np.number]).columns[0]

def safe_read_csv(path):
    try:
        return pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    except:
        return None

def scrape_and_save():
    """Logika scraping tetap sama, fokus pada keandalan data."""
    with st.spinner("Mengambil data terbaru dari bursa..."):
        raw = yf.download(TICKERS, start="2014-01-01", end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=False)
        if raw.empty: return False
        
        raw = raw.reset_index()
        saved = False
        for ticker in TICKERS:
            bank = ticker.replace(".JK", "")
            try:
                # Handle MultiIndex logic simply
                if isinstance(raw.columns, pd.MultiIndex):
                    df = pd.DataFrame({
                        'Date': raw['Date'],
                        'Adj Close': raw['Adj Close'][ticker] if 'Adj Close' in raw else raw['Close'][ticker],
                        'Volume': raw['Volume'][ticker]
                    })
                else:
                    # Fallback logic
                    df = pd.DataFrame({'Date': raw['Date'], 'Adj Close': raw['Adj Close'], 'Volume': raw['Volume']})
                
                df.dropna(inplace=True)
                df.set_index('Date', inplace=True)
                df.to_csv(CSV_FILES[bank])
                saved = True
            except: pass
        return saved

@st.cache_data
def load_data():
    datasets = {}
    full_df_list = []
    
    # Cek file, jika kosong scrape dulu
    if not os.path.exists(CSV_FILES['BBRI']):
        scrape_and_save()

    for bank, path in CSV_FILES.items():
        if os.path.exists(path):
            df = safe_read_csv(path)
            if df is not None:
                # Pre-calculation untuk performa
                pc = detect_price_col(df)
                df['Return'] = df[pc].pct_change()
                df['Vol_30'] = df['Return'].rolling(30).std() * np.sqrt(252)
                df['Cum_Ret'] = (1 + df['Return']).cumprod()
                df['Bank'] = bank
                df['Harga'] = df[pc]
                
                datasets[bank] = df
                full_df_list.append(df.reset_index())
                
    if full_df_list:
        return datasets, pd.concat(full_df_list)
    return {}, pd.DataFrame()

def get_zscore_outliers(df, col, thresh=3.0):
    s = df[col].dropna()
    z = np.abs(stats.zscore(s))
    return s[z > thresh]

# -------------------------
# 4. MAIN DASHBOARD UI
# -------------------------
datasets, df_full = load_data()

# Sidebar
with st.sidebar:
    st.header("🎛️ Kontrol")
    if st.button("Refresh Data"):
        scrape_and_save()
        load_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("Filter Tampilan")
    # Filter default: 1 tahun terakhir
    default_start = df_full['Date'].max() - pd.Timedelta(days=365)
    default_end = df_full['Date'].max()
    
    date_range = st.date_input("Rentang Waktu", [default_start, default_end])
    selected_banks = st.multiselect("Pilih Bank", list(BANK_COLORS.keys()), default=list(BANK_COLORS.keys()))

# Validasi Filter
if len(date_range) != 2 or not selected_banks:
    st.warning("Mohon pilih rentang tanggal dan bank.")
    st.stop()

# Filter Data untuk Tampilan (View Data)
mask = (df_full['Date'].dt.date >= date_range[0]) & (df_full['Date'].dt.date <= date_range[1]) & (df_full['Bank'].isin(selected_banks))
df_view = df_full[mask]

# === HEADER & KEY METRICS ===
st.title("Market Intelligence Dashboard")
st.markdown("Analisis performa saham bank BUMN. **Outlier dideteksi berdasarkan data historis penuh (2014-2024).**")
st.markdown("---")

# KPI Cards
cols = st.columns(len(selected_banks))
for idx, bank in enumerate(selected_banks):
    bank_data = df_view[df_view['Bank'] == bank]
    if bank_data.empty: continue
    
    curr_price = bank_data['Harga'].iloc[-1]
    start_price = bank_data['Harga'].iloc[0]
    change = (curr_price - start_price) / start_price
    
    with cols[idx]:
        st.markdown(f"<div style='border-top: 3px solid {BANK_COLORS[bank]}'></div>", unsafe_allow_html=True)
        st.metric(
            label=bank,
            value=f"Rp {curr_price:,.0f}",
            delta=f"{change:.2%}"
        )

st.markdown("---")

# === MAIN TABS ===
tab_trend, tab_risk, tab_outlier, tab_rec = st.tabs([
    "📈 Tren & Performa", 
    "⚖️ Risiko & Distribusi", 
    "🚨 Deteksi Outlier (Full History)", # Diganti dari 'Anomali'
    "💡 Rekomendasi"
])

# --- TAB 1: TREND ---
with tab_trend:
    col_main, col_stats = st.columns([3, 1])
    
    with col_main:
        # Grafik Utama
        fig_price = px.line(df_view, x='Date', y='Harga', color='Bank', color_discrete_map=BANK_COLORS)
        fig_price = style_chart(fig_price, title="Pergerakan Harga Saham (Periode Terpilih)", y_title="Harga (IDR)")
        fig_price.update_yaxes(tickprefix="Rp ")
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Grafik Pendukung
        st.markdown("#### Pertumbuhan Investasi")
        st.caption("Jika Anda menginvestasikan Rp 1 pada awal periode ini, inilah nilainya sekarang.")
        fig_cum = px.line(df_view, x='Date', y='Cum_Ret', color='Bank', color_discrete_map=BANK_COLORS)
        fig_cum = style_chart(fig_cum, y_title="Indeks Pertumbuhan (x)")
        fig_cum.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_cum, use_container_width=True)

    with col_stats:
        st.markdown("#### Statistik Periode Ini")
        summary_stats = df_view.groupby('Bank')['Harga'].agg(['min', 'max', 'mean'])
        st.dataframe(summary_stats.style.format("Rp {:,.0f}"), use_container_width=True)
        
        with st.expander("Lihat Data Mentah"):
            st.dataframe(df_view[['Date', 'Bank', 'Harga', 'Volume']], use_container_width=True)

# --- TAB 2: RISK & DISTRIBUTION ---
with tab_risk:
    c1, c2 = st.columns(2)
    
    with c1:
        # Volatilitas
        fig_vol = px.line(df_view, x='Date', y='Vol_30', color='Bank', color_discrete_map=BANK_COLORS)
        fig_vol = style_chart(fig_vol, title="Volatilitas Pasar (30-Day Rolling)", y_title="Tingkat Risiko")
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with c2:
        # Distribusi
        fig_box = px.box(df_view, x='Bank', y='Harga', color='Bank', color_discrete_map=BANK_COLORS)
        fig_box = style_chart(fig_box, title="Sebaran Harga (Boxplot)", y_title="Harga")
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.caption("Tips: Grafik volatilitas yang tinggi menunjukkan risiko jangka pendek yang lebih besar.")

# --- TAB 3: OUTLIER DETECTION (TERMINOLOGY UPDATED) ---
with tab_outlier:
    st.info("ℹ️ **Info Metodologi:** Analisis ini menggunakan Z-Score > 3.0 pada **SELURUH DATA HISTORIS** (bukan data terfilter) untuk menjaga integritas fakta sejarah.")
    
    zt = 3.0
    outlier_data = []
    
    # Hitung data outlier untuk ringkasan
    for bank in selected_banks:
        if bank not in datasets: continue
        df_full_bank = datasets[bank] # Gunakan data full
        outs = get_zscore_outliers(df_full_bank, 'Harga', thresh=zt)
        outlier_data.append({'Bank': bank, 'Jumlah Outlier': len(outs)})
    
    # Layout Balance: Kiri (Metrik), Kanan (Grafik)
    col_metrics, col_chart = st.columns([1, 3])
    
    with col_metrics:
        st.markdown("#### Ringkasan Outlier")
        st.dataframe(pd.DataFrame(outlier_data).set_index('Bank'), use_container_width=True)
        st.caption(f"Threshold Z-Score: {zt} (Fixed)")
        
    with col_chart:
        # Pilih bank untuk detail
        bank_choice = st.selectbox("Pilih Bank untuk Detail Visualisasi:", selected_banks)
        
        if bank_choice in datasets:
            df_viz = datasets[bank_choice]
            outliers = get_zscore_outliers(df_viz, 'Harga', thresh=zt)
            
            if len(outliers) > 0:
                fig_out = go.Figure()
                # Garis Harga
                fig_out.add_trace(go.Scatter(
                    x=df_viz.index, y=df_viz['Harga'], 
                    mode='lines', name='Harga Normal',
                    line=dict(color=BANK_COLORS[bank_choice], width=1.5), opacity=0.6
                ))
                # Titik Merah (Outlier)
                fig_out.add_trace(go.Scatter(
                    x=outliers.index, y=outliers.values,
                    mode='markers', name='OUTLIER', # Istilah diganti
                    marker=dict(color='#D6001C', size=8, symbol='x'),
                    hovertemplate="<b>OUTLIER TERDETEKSI</b><br>Tanggal: %{x|%Y-%m-%d}<br>Harga: Rp %{y:,.0f}"
                ))
                
                fig_out = style_chart(fig_out, title=f"Peta Persebaran Outlier: {bank_choice}", y_title="Harga")
                st.plotly_chart(fig_out, use_container_width=True)
                
                with st.expander("Lihat Detail Tanggal Outlier"):
                    st.dataframe(outliers.to_frame("Harga Outlier"), use_container_width=True)
            else:
                # Empty state
                st.success(f"✅ Tidak ditemukan outlier ekstrem pada saham {bank_choice} (Z-Score < 3.0).")

# --- TAB 4: RECOMMENDATION ---
with tab_rec:
    st.subheader("Kesimpulan & Rekomendasi")
    
    best_perf = df_view.groupby('Bank')['Return'].mean().idxmax()
    lowest_risk = df_view.groupby('Bank')['Vol_30'].mean().idxmin()
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.success(f"**Top Growth:** {best_perf}")
        st.caption("Bank dengan rata-rata return harian tertinggi pada periode ini.")
        
    with col_res2:
        st.info(f"**Paling Stabil:** {lowest_risk}")
        st.caption("Bank dengan rata-rata volatilitas terendah pada periode ini.")
        
    st.markdown("---")
    st.markdown("> **Catatan:** Analisis ini berdasarkan data teknikal historis semata. Keputusan investasi harus mempertimbangkan fundamental perusahaan dan profil risiko pribadi.")

st.markdown("---")
st.caption("Dashboard Version 3.1 (Professional UI - Terminology Fixed) | Data Source: Yahoo Finance")