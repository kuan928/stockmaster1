"""
台股進階分析系統 v3.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="台股進階分析系統 v3.0",
    page_icon="📈",
    layout="wide"
)

def get_taiwan_stock_data(stock_code, period="1y", interval="1d"):
    try:
        ticker = f"{stock_code}.TW"
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            ticker = f"{stock_code}.TWO"
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            return df, stock
        return None, None
    except Exception as e:
        st.error(f"資料抓取失敗: {str(e)}")
        return None, None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(high, low, close, k_period=9, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_bollinger_bands(series, period=20, std_dev=2):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def calculate_technical_indicators(df):
    df = df.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    df['RSI'] = calculate_rsi(df['Close'])
    k, d = calculate_stochastic(df['High'], df['Low'], df['Close'])
    df['K'] = k
    df['D'] = d
    upper, middle, lower = calculate_bollinger_bands(df['Close'])
    df['BB_upper'] = upper
    df['BB_middle'] = middle
    df['BB_lower'] = lower
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 100
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    return df

def detect_ma_crossover(df, lookback=5):
    if len(df) < lookback + 1:
        return {}
    crossovers = {
        'MA5_cross_MA10': {'status': None, 'days_ago': None},
        'MA10_cross_MA20': {'status': None, 'days_ago': None},
        'MA20_cross_MA60': {'status': None, 'days_ago': None}
    }
    for i in range(1, min(lookback + 1, len(df))):
        current = df.iloc[-i]
        previous = df.iloc[-i-1]
        if pd.notna(current['MA5']) and pd.notna(current['MA10']):
            if previous['MA5'] <= previous['MA10'] and current['MA5'] > current['MA10']:
                if crossovers['MA5_cross_MA10']['status'] is None:
                    crossovers['MA5_cross_MA10'] = {'status': 'golden', 'days_ago': i}
        if pd.notna(current['MA10']) and pd.notna(current['MA20']):
            if previous['MA10'] <= previous['MA20'] and current['MA10'] > current['MA20']:
                if crossovers['MA10_cross_MA20']['status'] is None:
                    crossovers['MA10_cross_MA20'] = {'status': 'golden', 'days_ago': i}
        if pd.notna(current['MA20']) and pd.notna(current['MA60']):
            if previous['MA20'] <= previous['MA60'] and current['MA20'] > current['MA60']:
                if crossovers['MA20_cross_MA60']['status'] is None:
                    crossovers['MA20_cross_MA60'] = {'status': 'golden', 'days_ago': i}
    return crossovers

def detect_bollinger_squeeze(df, lookback=20, threshold=3.0):
    if len(df) < lookback or 'BB_width' not in df.columns:
        return False, None, None
    recent_bb_width = df['BB_width'].iloc[-lookback:].dropna()
    if len(recent_bb_width) == 0:
        return False, None, None
    current_width = df['BB_width'].iloc[-1]
    avg_width = recent_bb_width.mean()
    min_width = recent_bb_width.min()
    is_squeezed = current_width < threshold or current_width == min_width
    return is_squeezed, current_width, avg_width

def check_bullish_setup(df):
    if len(df) < 60:
        return False, [], 0
    latest = df.iloc[-1]
    signals = []
    score = 0
    crossovers = detect_ma_crossover(df, lookback=10)
    if crossovers['MA5_cross_MA10']['status'] == 'golden':
        days = crossovers['MA5_cross_MA10']['days_ago']
        signals.append(f"✅ MA5黃金交叉MA10 ({days}天前)")
        score += 2
    if crossovers['MA10_cross_MA20']['status'] == 'golden':
        days = crossovers['MA10_cross_MA20']['days_ago']
        signals.append(f"✅ MA10黃金交叉MA20 ({days}天前)")
        score += 3
    if crossovers['MA20_cross_MA60']['status'] == 'golden':
        days = crossovers['MA20_cross_MA60']['days_ago']
        signals.append(f"✅ MA20黃金交叉MA60 ({days}天前)")
        score += 4
    is_squeezed, current_width, avg_width = detect_bollinger_squeeze(df)
    if is_squeezed and current_width is not None:
        signals.append(f"✅ 布林通道收縮 (寬度:{current_width:.2f}%)")
        score += 3
    if pd.notna(latest['MACD_hist']):
        if latest['MACD_hist'] > 0:
            signals.append(f"✅ MACD紅柱")
            score += 2
    if pd.notna(latest['MA20']):
        if latest['Close'] > latest['MA20']:
            signals.append(f"✅ 股價站上MA20")
            score += 2
    if pd.notna(latest['MA5']) and pd.notna(latest['MA10']) and pd.notna(latest['MA20']):
        if latest['Close'] > latest['MA5'] > latest['MA10'] > latest['MA20']:
            signals.append(f"⭐ 多頭排列!")
            score += 3
    if pd.notna(latest['Volume_MA5']):
        if latest['Volume'] > latest['Volume_MA5'] * 1.2:
            signals.append(f"✅ 成交量放大")
            score += 1
    is_bullish = score >= 6
    return is_bullish, signals, score

st.title("📈 台股進階分析系統 v3.0")
st.markdown("**均線交叉 × 布林收縮 × MACD轉強**")

tab1, tab2 = st.tabs(["🔍 單股分析", "📊 批次掃描"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        stock_code = st.text_input("輸入股票代號", value="2330")
    with col2:
        period = st.selectbox("時間範圍", ["3mo", "6mo", "1y", "2y"], index=1)
    
    if st.button("🚀 開始分析", type="primary"):
        with st.spinner(f"正在分析 {stock_code}..."):
            df, stock = get_taiwan_stock_data(stock_code, period=period)
            if df is not None and not df.empty:
                df = calculate_technical_indicators(df)
                try:
                    company_name = stock.info.get('longName', stock_code)
                except:
                    company_name = stock_code
                st.success(f"✅ {company_name} ({stock_code})")
                crossovers = detect_ma_crossover(df)
                is_bullish, signals, score = check_bullish_setup(df)
                is_squeezed, current_width, avg_width = detect_bollinger_squeeze(df)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if is_bullish:
                        st.success(f"✅ 多頭設定\n評分: {score}分")
                    else:
                        st.warning(f"⚠️ 觀察中\n評分: {score}分")
                with col2:
                    latest = df.iloc[-1]
                    macd_status = "紅柱" if latest['MACD_hist'] > 0 else "綠柱"
                    if latest['MACD_hist'] > 0:
                        st.success(f"MACD\n{macd_status}")
                    else:
                        st.error(f"MACD\n{macd_status}")
                with col3:
                    if is_squeezed and current_width:
                        st.success(f"布林收縮\n{current_width:.2f}%")
                    else:
                        st.info(f"布林寬度\n{current_width:.2f}%" if current_width else "N/A")
                if signals:
                    st.markdown("### 📋 詳細訊號")
                    for signal in signals:
                        st.success(signal) if '✅' in signal or '⭐' in signal else st.info(signal)
                st.markdown("### 📊 最新數據")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("收盤價", f"${latest['Close']:.2f}")
                    st.metric("MA20", f"${latest['MA20']:.2f}")
                with c2:
                    st.metric("RSI", f"{latest['RSI']:.2f}")
                    st.metric("ATR", f"{latest['ATR']:.2f}")
                with c3:
                    st.metric("K", f"{latest['K']:.2f}")
                    st.metric("D", f"{latest['D']:.2f}")
            else:
                st.error(f"❌ 無法取得 {stock_code} 的資料")

with tab2:
    st.markdown("## 📊 批次選股掃描")
    stock_list_input = st.text_area("輸入股票代號列表 (每行一個)", value="2330\n2317\n2454\n3008\n2308", height=200)
    col1, col2 = st.columns(2)
    with col1:
        scan_button = st.button("🚀 開始掃描", type="primary", use_container_width=True)
    with col2:
        min_score = st.slider("最低分數門檻", 4, 10, 6)
    
    if scan_button:
        stock_list = [s.strip() for s in stock_list_input.split('\n') if s.strip()]
        if stock_list:
            st.markdown(f"### 掃描結果 (共 {len(stock_list)} 檔)")
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, code in enumerate(stock_list):
                status_text.text(f"正在分析 {code}... ({i+1}/{len(stock_list)})")
                progress_bar.progress((i + 1) / len(stock_list))
                try:
                    df, stock = get_taiwan_stock_data(code, period="6mo")
                    if df is not None and len(df) >= 60:
                        df = calculate_technical_indicators(df)
                        is_bullish, signals, score = check_bullish_setup(df)
                        if score >= min_score:
                            latest = df.iloc[-1]
                            try:
                                company_name = stock.info.get('longName', code)
                            except:
                                company_name = code
                            results.append({
                                '代號': code,
                                '名稱': company_name,
                                '評分': score,
                                '收盤價': f"${latest['Close']:.2f}",
                                'RSI': f"{latest['RSI']:.1f}",
                                'MACD': '紅' if latest['MACD_hist'] > 0 else '綠',
                                '訊號數': len(signals)
                            })
                except:
                    continue
            status_text.empty()
            progress_bar.empty()
            if results:
                st.success(f"✅ 找到 {len(results)} 檔符合條件的股票!")
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values('評分', ascending=False)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                csv = df_results.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("📥 下載選股結果", data=csv, file_name=f"選股結果_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            else:
                st.warning("😢 沒有找到符合條件的股票")

st.markdown("---")
st.markdown("**台股進階分析系統 v3.0** | ⚠️ 投資有風險,請務必設定停損")
