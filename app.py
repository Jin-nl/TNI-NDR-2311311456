import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib
import yfinance as yf
import os

#Fetching NVDA stock data using yfinance
st.set_page_config(page_title="NVDA Stock Viewer", layout="wide")

# Font
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma', 'Arial', 'DejaVu Sans']

# Apply basic responsive style
st.markdown("""
    <style>
        @media (max-width: 768px) {
            h1, h2, h3, h4, h5, h6, .stMarkdown p {
                font-size: 90% !important;
            }
            .element-container { padding: 0.5rem !important; }
        }
        .stTable table {
            width: 100% !important;
            font-size: 0.85rem;
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
        .stTable td, .stTable th {
            padding: 0.25rem 0.5rem;
        }
        .stDataFrame div[data-testid="stHorizontalBlock"] {
            overflow-x: auto;
        }
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

#SlideBar
with st.sidebar:
    st.header("เลือกช่วงข้อมูล")
    selected_period = st.selectbox("ช่วงเวลา", ["7d", "1mo", "3mo", "6mo", "1y"], index=3)

# LOAD DATA
@st.cache_data
def load_data(period):
    nvda = yf.Ticker("NVDA")
    df = nvda.history(period=period).reset_index()
    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.to_excel("NVDA_Stocks_6M.xlsx", sheet_name="NVDA_Stocks_6M", index=False)
    return df.dropna()

# โหลดข้อมูล
df = load_data(selected_period)
df_sorted = df.sort_values("Date")

# แสดงวันที่ล่าสุด
latest_date = df["Date"].max().strftime("%Y-%m-%d")
st.sidebar.info(f"ข้อมูลล่าสุดถึงวันที่: **{latest_date}**")

# HEADER
with st.container():
    st.markdown("## NVDA (NVIDIA Corporation)")
    available_dates = df_sorted["Date"].dt.date.unique()
    selected_date = st.date_input("เลือกวันที่", value=available_dates[-1], min_value=available_dates[0], max_value=available_dates[-1])
    selected_row = df_sorted[df_sorted["Date"].dt.date == selected_date]

    if not selected_row.empty:
        selected_price = selected_row["Close"].values[0]
        selected_date_str = selected_date.strftime("%Y-%m-%d")
        current_index = df_sorted.index[df_sorted["Date"].dt.date == selected_date][0]
        if current_index > 0:
            previous_price = df_sorted.iloc[current_index - 1]["Close"]
            diff = selected_price - previous_price
            color = "green" if diff > 0 else "red" if diff < 0 else "gray"
            sign = "+" if diff > 0 else ""
            st.markdown(
                f"### ราคาปิดวันที่ {selected_date_str}: **${selected_price:,.2f}**  "
                f"<span style='color:{color}'>({sign}{diff:,.2f})</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"### ราคาปิดวันที่ {selected_date_str}: **${selected_price:,.2f}**")
    else:
        st.warning("ไม่มีข้อมูลของวันที่เลือก")

# Moving Averages Calculate
with st.sidebar:
    st.header("ตั้งค่า Moving Averages")
    sma_window = st.slider("SMA Window", min_value=5, max_value=60, value=20)
    ema_window = st.slider("EMA Window", min_value=5, max_value=60, value=20)


X = df_sorted["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = df_sorted["Close"].values
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
trend_poly = model_poly.predict(X_poly)
df_sorted["SMA"] = df_sorted["Close"].rolling(window=sma_window).mean()
df_sorted["EMA"] = df_sorted["Close"].ewm(span=ema_window, adjust=False).mean()

# PLOT SMA EMS
with st.container():
    st.subheader("กราฟราคาปิดและเส้นแนวโน้ม")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_sorted["Date"], df_sorted["Close"], label="ราคาปิด", color="#1f77b4", linewidth=2)
    ax.plot(df_sorted["Date"], trend_poly, label="Polynomial Trend", color="#2ca02c", linestyle="--", linewidth=2)
    if len(df_sorted) >= max(sma_window, ema_window):
        ax.plot(df_sorted["Date"], df_sorted["SMA"], label=f"SMA ({sma_window})", color="#ff7f0e", linewidth=1.8)
        ax.plot(df_sorted["Date"], df_sorted["EMA"], label=f"EMA ({ema_window})", color="#9467bd", linewidth=1.8)
    if not selected_row.empty:
        marker_date = selected_row["Date"].values[0]
        marker_price = selected_row["Close"].values[0]
        ax.scatter(marker_date, marker_price, color="red", s=80, zorder=5, label="Selected Date")
        ax.annotate(f"{marker_price:.2f} USD", (marker_date, marker_price), textcoords="offset points", xytext=(0,10), ha='center', color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.set_title("NVDA Stock Price and Trend", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    ax.tick_params(axis='x', labelrotation=45)
    st.pyplot(fig)

# DATA ช่วงวันที่ของข้อมูล
with st.container():
    st.markdown("---")
    st.subheader("ช่วงวันที่ของข้อมูล")
    col1, col2 = st.columns(2)
    col1.metric(label="เริ่มต้น", value=df_sorted['Date'].min().strftime('%Y-%m-%d'))
    col2.metric(label="สิ้นสุด", value=df_sorted['Date'].max().strftime('%Y-%m-%d'))

# TABS ผู้ถือหุ้นนี้
from streamlit_option_menu import option_menu
selected_tab = option_menu(
    menu_title=None,
    options=["ผู้ถือหุ้นนี้", "วิเคราะห์ทางเทคนิค"],
    icons=["people-fill", "bar-chart-fill"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#0e1117"},
        "icon": {"color": "#ffffff", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "--hover-color": "#262730"},
        "nav-link-selected": {"background-color": "#1f77b4"},
    }
)

if selected_tab == "ผู้ถือหุ้นนี้":
    top_shareholders = pd.DataFrame({
        "ผู้ถือหุ้น": [
            "The Vanguard Group Inc.", "BlackRock Inc.", "FMR LLC (Fidelity)", "State Street Corporation",
            "Jensen Huang (CEO)", "Geode Capital Management LLC", "T. Rowe Price Associates Inc.",
            "JPMorgan Chase & Co.", "Norges Bank Inv. Mgmt", "Morgan Stanley"
        ],
        "ประเภทผู้ถือหุ้น": ["สถาบัน", "สถาบัน", "กองทุน", "สถาบัน", "บุคคลทั่วไป", "สถาบัน", "กองทุน", "สถาบัน", "กองทุน", "สถาบัน"],
        "จำนวนหุ้น (พันล้านหุ้น)": [2.19, 1.90, 1.30, 0.95, 0.93, 0.50, 0.49, 0.35, 0.32, 0.30],
        "สัดส่วนการถือครอง (%)": [8.97, 7.78, 5.30, 3.90, 3.79, 2.05, 2.01, 1.43, 1.33, 1.23],
        "มูลค่าการถือครอง (พันล้าน USD)": [287.92, 250.00, 170.00, 125.00, 123.00, 66.00, 65.00, 45.00, 42.00, 40.00],
        "แนวโน้มการถือครอง": [
            "🔼 เพิ่มขึ้น 1.2M", "➡ คงที่", "🔽 ลดลง 0.8M", "➡ คงที่", "🔼 เพิ่มขึ้น 0.4M",
            "➡ คงที่", "🔼 เพิ่มขึ้น 0.5M", "🔽 ลดลง 0.2M", "➡ คงที่", "🔼 เพิ่มขึ้น 0.1M"
        ]
    })
    top_shareholders.reset_index(drop=True, inplace=True)
    top_shareholders.index += 1
    top_shareholders.index.name = "อันดับ"
    top_shareholders["จำนวนหุ้น (พันล้านหุ้น)"] = top_shareholders["จำนวนหุ้น (พันล้านหุ้น)"].map(lambda x: f"{x:,.2f}")
    top_shareholders["สัดส่วนการถือครอง (%)"] = top_shareholders["สัดส่วนการถือครอง (%)"].map(lambda x: f"{x:.2f}%")
    top_shareholders["มูลค่าการถือครอง (พันล้าน USD)"] = top_shareholders["มูลค่าการถือครอง (พันล้าน USD)"].map(lambda x: f"${x:,.2f}")
    st.subheader("อันดับผู้ถือหุ้นรายใหญ่ของ NVDA (NVIDIA)")
    st.table(top_shareholders)

elif selected_tab == "วิเคราะห์ทางเทคนิค":
    st.subheader("RSI (Relative Strength Index)")
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    df_sorted["RSI"] = compute_rsi(df_sorted["Close"])
    fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
    ax_rsi.plot(df_sorted["Date"], df_sorted["RSI"], label="RSI", color="orange")
    ax_rsi.axhline(70, linestyle="--", color="red", alpha=0.7, label="Overbought (70)")
    ax_rsi.axhline(30, linestyle="--", color="green", alpha=0.7, label="Oversold (30)")
    ax_rsi.set_title("RSI Indicator (14 วัน)")
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True, linestyle="--", alpha=0.3)
    ax_rsi.legend()
    st.pyplot(fig_rsi)
    st.caption("ค่า RSI เหนือ 70 → อาจซื้อมากเกินไป<br>ต่ำกว่า 30 → อาจขายมากเกินไป", unsafe_allow_html=True)

    st.subheader("MACD (Moving Average Convergence Divergence)")
    short_ema = df_sorted["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df_sorted["Close"].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    fig_macd, ax_macd = plt.subplots(figsize=(12, 3))
    ax_macd.plot(df_sorted["Date"], macd_line, label="MACD", color="#00bfff")
    ax_macd.plot(df_sorted["Date"], signal_line, label="Signal Line", color="purple")
    ax_macd.set_title("MACD Indicator")
    ax_macd.grid(True, linestyle="--", alpha=0.3)
    ax_macd.legend()
    st.pyplot(fig_macd)

# FOOTER
st.markdown("---")
st.caption("""
    NVDA Stock Analysis Dashboard เว็บไซต์นี้แสดงข้อมูลราคาปิดของหุ้น NVIDIA ย้อนหลัง 6 เดือน  
    พร้อมเส้นแนวโน้ม Polynomial Regression และค่าเฉลี่ยเคลื่อนที่ (SMA/EMA)
""")
st.markdown("---")
st.caption("2311311456 ศรันย์ เขมะสุข")