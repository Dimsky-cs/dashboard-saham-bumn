# app.py — Unified Dashboard (FINAL: USER CODE BASE + BEAUTIFUL TAB 4)
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
from scipy import stats
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
    'BBNI': "#005E6A", # Tosca BNI (Digelapkan sedikit agar kontras di background putih)
    'BMRI': '#F3A800', # Mandiri Emas
    'BBTN': '#EC1C24'  # BTN Merah
}

# -------------------------
# HELPERS
# -------------------------
def flatten_multiindex_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Meratakan kolom MultiIndex (jika ada) menjadi satu string."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join([str(x) for x in col]).strip() for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def clean_chart_layout(fig, title="", y_title=""):
    """Menerapkan prinsip Data-Ink Ratio: Hapus grid berlebih, perjelas label."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        hovermode="x unified",
        height=400,
        margin=dict(t=40, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="")
    )
    fig.update_xaxes(showgrid=False, title="") # Hapus grid vertikal (Visual Noise)
    fig.update_yaxes(showgrid=True, gridcolor='#eee', title=y_title)
    return fig

def detect_price_col(df: pd.DataFrame) -> Optional[str]:
    """Mendeteksi kolom harga utama (Adj Close, lalu Close, lalu numerik pertama)."""
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
    """Membaca CSV dengan aman, mencoba parse 'Date' sebagai index."""
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
    """Download dari yfinance dan simpan per-bank CSV + master CSV."""
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
                tmp.columns = [str(c) for c in tmp.columns]
                def find_col(pref):
                    pref_s = pref.replace(" ", "").lower()
                    for c in tmp.columns:
                        if pref_s in c.lower().replace(" ", "") and bank.lower() in c.lower():
                            return c
                    for c in tmp.columns:
                        if pref_s in c.lower().replace(" ", ""):
                            return c
                    return None
                df = pd.DataFrame({
                    "Date": tmp["Date"],
                    "Open": tmp.get(find_col("open") or "Open"),
                    "High": tmp.get(find_col("high") or "High"),
                    "Low": tmp.get(find_col("low") or "Low"),
                    "Close": tmp.get(find_col("close") or "Close"),
                    "Adj Close": tmp.get(find_col("adj close") or find_col("adjclose") or "Adj Close"),
                    "Volume": tmp.get(find_col("volume") or "Volume"),
                })
            
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
    else:
        st.error("Gagal membuat master CSV.")
        return False

# -------------------------
# LOAD CSV (cached)
# -------------------------
@st.cache_data
def load_all_csvs() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Memuat semua CSV individu ke dict dan CSV gabungan ke DataFrame."""
    datasets = {}
    all_rows = []
    for bank, path in CSV_FILES.items():
        if os.path.exists(path):
            df = safe_read_csv(path)
            if df is not None:
                df = df.dropna()          # Buang baris kosong
                df = df.drop_duplicates() # Buang data ganda
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
    """Menghitung semua metrik (Return, Vol, CumReturn) SATU KALI."""
    metrics_list = []
    for bank, dfb in _datasets.items():
        if dfb is None or dfb.empty:
            continue
            
        pc = detect_price_col(dfb)
        if pc is None:
            continue
            
        df = dfb.copy()
        df[pc] = pd.to_numeric(df[pc], errors='coerce')
        df = df.dropna(subset=[pc])
        
        df['Return'] = df[pc].pct_change()
        df['Volatility_30d'] = df['Return'].rolling(window=30).std() * np.sqrt(252)
        df['Cumulative_Return'] = (1 + df['Return']).cumprod()
        df['Daily_Return_Pct'] = df['Return'] * 100
        df['Harga'] = df[pc]
        
        df = df.dropna(subset=['Return', 'Cumulative_Return', 'Daily_Return_Pct'])
        df = df.reset_index()
        df['Bank'] = bank
        
        metrics_list.append(df)
    
    if not metrics_list:
        return pd.DataFrame()
    
    return pd.concat(metrics_list, ignore_index=True)

# -------------------------
# OUTLIER DETECTION
# -------------------------
def outliers_zscore(series: pd.Series, thresh: float = 3.0) -> pd.Series:
    """Mendeteksi outlier menggunakan Z-score."""
    s = series.dropna().astype(float)
    if len(s) < 2:
        return pd.Series(dtype=float)
    z = np.abs(stats.zscore(s))
    return s[z > thresh]

# -------------------------
# UI: Sidebar
# -------------------------
# st.sidebar.title("Kontrol Data")
# if st.sidebar.button("Update Data (Scrape Ulang dari Yahoo)"):
#     ok = scrape_and_save()
#     if ok:
#         load_all_csvs.clear()
#         get_all_metrics.clear()
#         st.rerun()

st.sidebar.write("Data folder:", DATA_FOLDER)
st.sidebar.write("Tickers:", ", ".join(TICKERS))
# st.sidebar.caption("Mode: Hybrid (Load CSV, scrape jika perlu)")

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
    st.error("Tidak ada data untuk divisualisasikan. Pastikan CSV tersedia atau lakukan Update Data.")
    st.stop()
    
df_metrics = get_all_metrics(datasets)

# -------------------------
# LAYOUT: Judul, Filter Interaktif & KPI Dinamis
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
    
    date_range = st.date_input(
        "Pilih rentang tanggal", 
        [date_min_limit, date_max_limit], 
        min_value=date_min_limit, 
        max_value=date_max_limit
    )

if len(date_range) != 2:
    st.warning("Silakan pilih rentang tanggal yang valid (awal dan akhir).")
    st.stop()

date_mask_raw = (combined_raw['Date'].dt.date >= date_range[0]) & (combined_raw['Date'].dt.date <= date_range[1])
bank_mask_raw = combined_raw['Bank'].isin(selected_banks)
df_filtered_raw = combined_raw[bank_mask_raw & date_mask_raw].copy()

df_metrics['Date'] = pd.to_datetime(df_metrics['Date'])
date_mask_metrics = (df_metrics['Date'].dt.date >= date_range[0]) & (df_metrics['Date'].dt.date <= date_range[1])
bank_mask_metrics = df_metrics['Bank'].isin(selected_banks)
df_filtered_metrics = df_metrics[bank_mask_metrics & date_mask_metrics].copy()

if df_filtered_raw.empty or df_filtered_metrics.empty:
    st.warning("Tidak ada data di rentang dan pemilihan bank yang diberikan.")
    st.stop()
    
st.markdown("---")
total_rows_filtered = f"{len(df_filtered_raw):,}"
st.metric(label="📄 Total Baris Data (Hasil Filter)", value=total_rows_filtered)
st.markdown("---")

# -------------------------
# TABS: EDA / Charts / Outliers / Extreme / Recommendation
# -------------------------
# NAMA TAB DIKEMBALIKAN SESUAI KODE AWAL ANDA
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & EDA", 
    "📈 Charts (Trend/Vol/Returns)", 
    "📦 Hist", 
    "🚨 Outliers & Extreme", 
    "💡 Recommendation"
])

with tab1:
    st.subheader("Sample Data Mentah (filtered)")
    st.dataframe(df_filtered_raw.head(200))
    st.markdown("*Descriptive statistics (numeric)*")
    st.dataframe(df_filtered_raw.select_dtypes(include=[np.number]).describe().T)

with tab2:
    st.subheader("Trend Harga — multi-bank")
    price_col = 'Adj Close' if 'Adj Close' in df_filtered_raw.columns else ('Close' if 'Close' in df_filtered_raw.columns else detect_price_col(df_filtered_raw))
    
    fig_price = px.line(
        df_filtered_raw, 
        x='Date', 
        y=price_col, 
        color='Bank', 
        title=f"Tren Harga ({price_col})",
        color_discrete_map=BANK_COLORS
    )
    fig_price.update_layout(
        hovermode="x unified", 
        template="plotly_white", 
        height=520,
        xaxis_title="Tahun",
        yaxis_title="Harga (IDR)"
    )
    fig_price.update_traces(
        hovertemplate="Tanggal: %{x|%Y-%m-%d}<br>" +
                      "Harga: Rp %{y:,.0f}<extra></extra>"
    )
    fig_price.update_yaxes(tickprefix="Rp ")
    fig_price.update_xaxes(dtick="M12", tickformat="%Y")
    
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("---")
    st.subheader("Analisis Harga, & Cumulative Return")
    
    if not df_filtered_metrics.empty:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=("Harga (Adj Close)", "Cumulative Return (Index)"))
        
        for bank in selected_banks:
            d = df_filtered_metrics[df_filtered_metrics['Bank']==bank]
            if d.empty: continue
            
            bank_color = BANK_COLORS.get(bank, None)

            fig.add_trace(go.Scatter(
                x=d['Date'], y=d['Harga'], mode='lines', name=bank, 
                line=dict(color=bank_color),
                hovertemplate="Tanggal: %{x|%Y-%m-%d}<br>Harga: Rp %{y:,.0f}<extra></extra>"
            ), row=1, col=1)
            
            # fig.add_trace(go.Scatter(
            #     x=d['Date'], y=d['Volatility_30d'], mode='lines', name=bank, showlegend=False, 
            #     line=dict(color=bank_color),
            #     hovertemplate="Tanggal: %{x|%Y-%m-%d}<br>Volatility: %{y:.3f}<extra></extra>"
            # ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=d['Date'], y=d['Cumulative_Return'], mode='lines', name=bank, showlegend=False, 
                line=dict(color=bank_color),
                hovertemplate="Tanggal: %{x|%Y-%m-%d}<br>Cumulative: %{y:.2f}x<extra></extra>"
            ), row=2, col=1)
            
        fig.update_yaxes(title_text="Harga (IDR)", row=1, tickprefix="Rp ")
        # fig.update_yaxes(title_text="Volatility", row=2)
        fig.update_yaxes(title_text="Cumulative Index", row=2)
        
        fig.update_layout(
            height=900, 
            hovermode="x unified", 
            template="plotly_white",
            xaxis2_showticklabels=True,
            xaxis2_dtick="M12",
            xaxis2_tickformat="%Y",
            xaxis2_title="Tahun"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada data metrik yang bisa dihitung untuk bank terpilih.")

with tab3:
    st.subheader("Histogram (per fitur)")
    cols_to_plot = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    available_cols = [c for c in cols_to_plot if c in df_filtered_raw.columns]
    
    if not available_cols:
        st.info("Kolom numeric tidak tersedia.")
    else:
        for col in available_cols:
            st.markdown(f"*Distribusi: {col}*")
            fig = px.histogram(
                df_filtered_raw, 
                x=col, 
                color='Bank', 
                barmode='overlay', 
                nbins=60,
                color_discrete_map=BANK_COLORS
            )
            st.plotly_chart(fig, use_container_width=True)

        
with tab4:
    st.subheader("Outlier detection (Z-score)")
    
    # === KONFIGURASI SMART THRESHOLD ===
    # BBTN dikasih 2.8 agar sejarah 22 outlier-nya muncul
    # Bank lain (BBRI, BBNI, BMRI) tetap 3.0 agar tidak muncul noise (tetap 0)
    THRESHOLDS = {
        'BBTN': 3.0,
        'BBRI': 3.0, 
        'BBNI': 3.0, 
        'BMRI': 3.0
    }
    
    st.info("ℹ️ Analisis outlier menggunakan Z-Score 3 pada seluruh data.")
    
    outlier_summary = []
    
    for bank in selected_banks: 
        if bank not in datasets: continue
        
        # 1. Ambil Data Penuh
        df_full = datasets[bank].copy()
        pc = detect_price_col(df_full)
        if pc is None: continue

        df_full_raw = datasets[bank].copy()
        df_full[pc] = pd.to_numeric(df_full[pc], errors='coerce')
        df_clean = df_full.dropna(subset=[pc])
        
        # 2. Tentukan Threshold Khusus Bank ini
        # Jika bank ada di daftar THRESHOLDS, pakai nilainya. Jika tidak, pakai 3.0
        current_zt = THRESHOLDS.get(bank, 3.0)
        
        # 3. Hitung
        outs = outliers_zscore(df_clean[pc], thresh=current_zt)
        
        # Simpan ringkasan
        outlier_summary.append({
            "Bank": bank, "PriceCol": pc, "Outliers": len(outs), "Total Data": len(df_clean)
        })
        
        # 4. Tampilkan Judul & Layout
        st.markdown(f"### {bank}")
        
        col_metric, col_viz = st.columns([1, 3])
        
        with col_metric:
            st.metric(
                label="Jumlah Kejadian Outlier",
                value=f"{len(outs)}",
                delta="Kejadian Ekstrem" if len(outs) > 0 else "Normal / Stabil",
                delta_color="inverse" if len(outs) > 0 else "normal"
            )
            st.caption(f"Z-Score Threshold: {current_zt}") # Tampilkan threshold yang dipakai
            
            if len(outs) > 0:
                with st.expander("Lihat Tanggal Kejadian"):
                    st.dataframe(outs.to_frame("Harga").sort_index(), height=250)
            else:
                st.success(f"✅ Pergerakan harga stabil.")
        
        with col_viz:
            # Grafik selalu dibuat
            fig_out = go.Figure()
            color = BANK_COLORS.get(bank, 'blue')
            
            # Garis Harga
            fig_out.add_trace(go.Scatter(
                x=df_clean.index, y=df_clean[pc], 
                mode='lines', name='Harga Normal',
                line=dict(color=color, width=1.5), opacity=0.8,
                hovertemplate="Tanggal: %{x|%Y-%m-%d}<br>Harga: Rp %{y:,.0f}<extra></extra>"
            ))
            
            # Titik Outlier
            if len(outs) > 0:
                fig_out.add_trace(go.Scatter(
                    x=outs.index, y=outs.values, 
                    mode='markers', name='OUTLIER',
                    marker=dict(color='#D6001C', size=8, symbol='x'),
                    hovertemplate="<b>OUTLIER</b><br>Tanggal: %{x|%Y-%m-%d}<br>Harga: Rp %{y:,.0f}<extra></extra>"
                ))
                title_text = f"Peta Sebaran Outlier: {bank}"
            else:
                title_text = f"Tren Harga Historis: {bank}"
            
            fig_out = clean_chart_layout(fig_out, title=title_text, y_title="Harga (IDR)")
            st.plotly_chart(fig_out, use_container_width=True)
        
        st.markdown("---")

    # ... (Bagian Extreme movements tetap sama) ...
    st.subheader("Rekor Kenaikan & Penurunan Harian Terbesar")
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
            "Kenaikan Tertinggi (Spike)": f"{d_raw.loc[idxmax, 'Ret']:.2f}% pada {idxmax.strftime('%Y-%m-%d')}",
            "Penurunan Terdalam (Drop)": f"{d_raw.loc[idxmin, 'Ret']:.2f}% pada {idxmin.strftime('%Y-%m-%d')}"
        })
    
    st.table(pd.DataFrame(extremes).set_index('Bank'))

# with tab4:
#     st.subheader("Outlier detection (Z-score)")
#     zt = 3
#     st.info(f"Analisis dilakukan pada SELURUH data historis (2014-2024) dengan threshold Z-score {zt}.")

#     outlier_summary = []
    
#     for bank in selected_banks:
#         if bank not in datasets: continue
        
#         # 1. Data Prep (Full History)
#         df_full = datasets[bank].copy()
#         pc = detect_price_col(df_full)
#         if pc is None: continue
        
#         df_full[pc] = pd.to_numeric(df_full[pc], errors='coerce')
#         df_clean = df_full.dropna(subset=[pc])
#         outs = outliers_zscore(df_clean[pc], thresh=zt)
        
#         # 2. Judul Bank
#         st.markdown(f"### {bank}")
        
#         # 3. Layout 2 Kolom (Balance)
#         col_left, col_right = st.columns([1, 3])
        
#         # --- KOLOM KIRI: METRIK & DATA ---
#         with col_left:
#             st.metric(
#                 label="Jumlah Outlier", 
#                 value=f"{len(outs)}", 
#                 delta="Terdeteksi" if len(outs) > 0 else "Aman",
#                 delta_color="inverse"
#             )
#             st.caption(f"Total Data: {len(df_clean)}")
            
#             if len(outs) > 0:
#                 with st.expander("Lihat Data"):
#                     st.dataframe(outs.to_frame("Harga"), height=200)
#             else:
#                 st.success("Data stabil.")

#         # --- KOLOM KANAN: GRAFIK BERSIH (Data-Ink Ratio) ---
#         with col_right:
#             # Grafik selalu dibuat agar tidak kosong
#             fig_out = go.Figure()
#             color = BANK_COLORS.get(bank, 'blue')
            
#             # Garis Harga (Tipis)
#             fig_out.add_trace(go.Scatter(
#                 x=df_clean.index, y=df_clean[pc], 
#                 mode='lines', name='Harga',
#                 line=dict(color=color, width=1), opacity=0.6,
#                 hovertemplate="Tanggal: %{x|%Y-%m-%d}<br>Harga: Rp %{y:,.0f}<extra></extra>"
#             ))
            
#             # Titik Outlier (Jika Ada)
#             if len(outs) > 0:
#                 fig_out.add_trace(go.Scatter(
#                     x=outs.index, y=outs.values, 
#                     mode='markers', name='OUTLIER',
#                     marker=dict(color='#D6001C', size=6, symbol='x'),
#                     hovertemplate="<b>OUTLIER</b><br>Tanggal: %{x|%Y-%m-%d}<br>Harga: Rp %{y:,.0f}<extra></extra>"
#                 ))
#                 title_text = f"Peta Outlier: {bank}"
#             else:
#                 title_text = f"Tren Harga: {bank} (Stabil)"

#             # Gunakan helper untuk bersihkan visual noise
#             fig_out = clean_chart_layout(fig_out, title=title_text, y_title="Harga (IDR)")
#             st.plotly_chart(fig_out, use_container_width=True)
        
#         st.markdown("---")

#     st.subheader("Extreme movements (max spike / max drop per bank)")
    
#     extremes = []
#     for bank in selected_banks:
#         if bank not in datasets: continue
#         d_raw = datasets[bank].copy()
#         pc = detect_price_col(d_raw)
#         d_raw[pc] = pd.to_numeric(d_raw[pc], errors='coerce')
#         d_raw['Ret'] = d_raw[pc].pct_change() * 100
#         d_raw = d_raw.dropna(subset=['Ret'])
        
#         idxmax = d_raw['Ret'].idxmax()
#         idxmin = d_raw['Ret'].idxmin()
        
#         extremes.append({
#             "Bank": bank,
#             "Max Spike": f"{d_raw.loc[idxmax, 'Ret']:.2f}% ({idxmax.strftime('%Y-%m-%d')})",
#             "Max Drop": f"{d_raw.loc[idxmin, 'Ret']:.2f}% ({idxmin.strftime('%Y-%m-%d')})"
#         })
    
#     st.table(pd.DataFrame(extremes).set_index('Bank'))

with tab5:
    st.subheader("Automated Recommendation (simple rule-based)")
    
    stats = []
    for bank in selected_banks:
        dfb = df_filtered_metrics[df_filtered_metrics['Bank'] == bank]
        if dfb.empty: continue
        
        start_price = dfb.iloc[0]['Harga']
        end_price = dfb.iloc[-1]['Harga']
        
        total_return = (end_price / start_price) - 1
        annual_vol = dfb['Return'].std() * np.sqrt(252)
        
        stats.append({'Bank': bank, 'TotalReturn': total_return, 'AnnualVol': annual_vol})
        
    if not stats:
        st.info("Tidak ada statistik untuk rekomendasi.")
    else:
        sdf = pd.DataFrame(stats).sort_values('TotalReturn', ascending=False).reset_index(drop=True)
        sdf['TotalReturn%'] = (sdf['TotalReturn'] * 100).round(2).astype(str) + '%'
        sdf['AnnualVol%'] = (sdf['AnnualVol'] * 100).round(2).astype(str) + '%'
        
        st.dataframe(sdf[['Bank','TotalReturn%','AnnualVol%']])
        
        best = sdf.iloc[0]['Bank']; worst = sdf.iloc[-1]['Bank']
        
        st.markdown(f"- *Growth pick:* **{best}** (highest total return in selected range)")
        st.markdown(f"- *Avoid (low return / high risk):* **{worst}** (lowest total return in selected range)")
        st.markdown("> Catatan: rekomendasi sederhana berdasarkan rentang tanggal; gunakan analisis fundamental & risk management.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Dashboard app: Adapted from Colab visualizations.")