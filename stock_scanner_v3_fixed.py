"""
台股進階分析系統 v3.0 - 增強版
新增功能:
1. 均線交叉偵測 (5x10, 10x20, 20x60)
2. 布林通道收縮偵測
3. MACD轉強訊號
4. 綜合選股篩選器
5. 批次掃描多檔股票
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

# 嘗試導入 pandas_ta,如果失敗則提示
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    st.error("❌ 缺少 pandas-ta 套件,請確認 requirements.txt 包含 pandas-ta")
    st.stop()

# 頁面設定
st.set_page_config(
    page_title="台股進階分析系統 v3.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        font-size: 18px;
        font-weight: 600;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    .signal-card {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 0.5rem 0;
        background: white;
    }
    .signal-buy {
        border-color: #00C853;
        background: #E8F5E9;
    }
    .signal-sell {
        border-color: #FF1744;
        background: #FFEBEE;
    }
    .signal-neutral {
        border-color: #FFA726;
        background: #FFF3E0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 工具函數 ====================

@st.cache_data(ttl=60)
def get_taiwan_stock_data(stock_code, period="1y", interval="1d"):
    """抓取台股資料"""
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
        else:
            return None, None
            
    except Exception as e:
        st.error(f"資料抓取失敗: {str(e)}")
        return None, None

def calculate_technical_indicators(df):
    """計算技術指標 - 使用 pandas_ta"""
    try:
        df = df.copy()
        
        # 移動平均線
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # MACD - 使用 pandas_ta
        macd_result = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd_result is not None and not macd_result.empty:
            df['MACD'] = macd_result['MACD_12_26_9']
            df['MACD_signal'] = macd_result['MACDs_12_26_9']
            df['MACD_hist'] = macd_result['MACDh_12_26_9']
        
        # RSI - 使用 pandas_ta
        rsi_result = ta.rsi(df['Close'], length=14)
        if rsi_result is not None:
            df['RSI'] = rsi_result
        
        # KD (Stochastic) - 使用 pandas_ta
        stoch_result = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
        if stoch_result is not None and not stoch_result.empty:
            df['K'] = stoch_result['STOCHk_9_3_3']
            df['D'] = stoch_result['STOCHd_9_3_3']
        
        # 布林通道 - 使用 pandas_ta
        bbands_result = ta.bbands(df['Close'], length=20, std=2)
        if bbands_result is not None and not bbands_result.empty:
            df['BB_upper'] = bbands_result['BBU_20_2.0']
            df['BB_middle'] = bbands_result['BBM_20_2.0']
            df['BB_lower'] = bbands_result['BBL_20_2.0']
        
        # 布林寬度 (用於判斷收縮)
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns and 'BB_middle' in df.columns:
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 100
        
        # ATR - 使用 pandas_ta
        atr_result = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        if atr_result is not None:
            df['ATR'] = atr_result
        
        # 成交量移動平均
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        
        return df
    except Exception as e:
        st.error(f"計算技術指標時發生錯誤: {str(e)}")
        return df

def detect_ma_crossover(df, lookback=5):
    """
    偵測均線交叉
    lookback: 往前看幾根K棒
    """
    if len(df) < lookback + 1:
        return {}
    
    crossovers = {
        'MA5_cross_MA10': {'status': None, 'days_ago': None},
        'MA10_cross_MA20': {'status': None, 'days_ago': None},
        'MA20_cross_MA60': {'status': None, 'days_ago': None}
    }
    
    # 檢查最近 lookback 天內的交叉
    for i in range(1, min(lookback + 1, len(df))):
        current = df.iloc[-i]
        previous = df.iloc[-i-1]
        
        # MA5 x MA10
        if pd.notna(current['MA5']) and pd.notna(current['MA10']):
            if previous['MA5'] <= previous['MA10'] and current['MA5'] > current['MA10']:
                if crossovers['MA5_cross_MA10']['status'] is None:
                    crossovers['MA5_cross_MA10'] = {'status': 'golden', 'days_ago': i}
            elif previous['MA5'] >= previous['MA10'] and current['MA5'] < current['MA10']:
                if crossovers['MA5_cross_MA10']['status'] is None:
                    crossovers['MA5_cross_MA10'] = {'status': 'death', 'days_ago': i}
        
        # MA10 x MA20
        if pd.notna(current['MA10']) and pd.notna(current['MA20']):
            if previous['MA10'] <= previous['MA20'] and current['MA10'] > current['MA20']:
                if crossovers['MA10_cross_MA20']['status'] is None:
                    crossovers['MA10_cross_MA20'] = {'status': 'golden', 'days_ago': i}
            elif previous['MA10'] >= previous['MA20'] and current['MA10'] < current['MA20']:
                if crossovers['MA10_cross_MA20']['status'] is None:
                    crossovers['MA10_cross_MA20'] = {'status': 'death', 'days_ago': i}
        
        # MA20 x MA60
        if pd.notna(current['MA20']) and pd.notna(current['MA60']):
            if previous['MA20'] <= previous['MA60'] and current['MA20'] > current['MA60']:
                if crossovers['MA20_cross_MA60']['status'] is None:
                    crossovers['MA20_cross_MA60'] = {'status': 'golden', 'days_ago': i}
            elif previous['MA20'] >= previous['MA60'] and current['MA20'] < current['MA60']:
                if crossovers['MA20_cross_MA60']['status'] is None:
                    crossovers['MA20_cross_MA60'] = {'status': 'death', 'days_ago': i}
    
    return crossovers

def detect_bollinger_squeeze(df, lookback=20, threshold=3.0):
    """
    偵測布林通道收縮
    threshold: 布林寬度百分比閾值 (越小代表越收縮)
    """
    if len(df) < lookback or 'BB_width' not in df.columns:
        return False, None, None
    
    recent_bb_width = df['BB_width'].iloc[-lookback:].dropna()
    
    if len(recent_bb_width) == 0:
        return False, None, None
    
    current_width = df['BB_width'].iloc[-1]
    avg_width = recent_bb_width.mean()
    min_width = recent_bb_width.min()
    
    # 判斷是否處於收縮狀態
    is_squeezed = current_width < threshold or current_width == min_width
    
    return is_squeezed, current_width, avg_width

def check_bullish_setup(df):
    """
    檢查多頭設定
    條件:
    1. 近期有均線黃金交叉
    2. 布林通道收縮
    3. MACD紅柱 或 即將黃金交叉
    4. 股價站上關鍵均線
    """
    if len(df) < 60:
        return False, []
    
    latest = df.iloc[-1]
    signals = []
    score = 0
    
    # 1. 檢查均線交叉
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
    
    # 2. 檢查布林收縮
    is_squeezed, current_width, avg_width = detect_bollinger_squeeze(df)
    if is_squeezed and current_width is not None:
        signals.append(f"✅ 布林通道收縮 (寬度:{current_width:.2f}%)")
        score += 3
    
    # 3. 檢查MACD
    if 'MACD_hist' in df.columns and pd.notna(latest['MACD_hist']):
        if latest['MACD_hist'] > 0:
            signals.append(f"✅ MACD紅柱 ({latest['MACD_hist']:.3f})")
            score += 2
        elif 'MACD' in df.columns and 'MACD_signal' in df.columns:
            if latest['MACD_hist'] > -0.5 and latest['MACD'] > latest['MACD_signal']:
                signals.append(f"⚠️ MACD即將黃金交叉")
                score += 1
    
    # 檢查MACD是否剛黃金交叉
    if len(df) >= 2 and 'MACD' in df.columns and 'MACD_signal' in df.columns:
        prev = df.iloc[-2]
        if pd.notna(prev['MACD']) and pd.notna(latest['MACD']):
            if prev['MACD'] <= prev['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
                signals.append(f"⭐ MACD剛黃金交叉!")
                score += 3
    
    # 4. 檢查股價位置
    if pd.notna(latest['MA20']):
        if latest['Close'] > latest['MA20']:
            signals.append(f"✅ 股價站上MA20")
            score += 2
    
    if pd.notna(latest['MA5']) and pd.notna(latest['MA10']) and pd.notna(latest['MA20']):
        if latest['Close'] > latest['MA5'] > latest['MA10'] > latest['MA20']:
            signals.append(f"⭐ 多頭排列!")
            score += 3
    
    # 5. 檢查成交量
    if pd.notna(latest['Volume_MA5']):
        if latest['Volume'] > latest['Volume_MA5'] * 1.2:
            signals.append(f"✅ 成交量放大")
            score += 1
    
    # 總分 >= 6 分視為符合多頭設定
    is_bullish = score >= 6
    
    return is_bullish, signals, score

def plot_candlestick_with_signals(df, stock_code, crossovers):
    """繪製K線圖並標示訊號"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(f'{stock_code} K線圖與均線交叉', '成交量', 'MACD', 'RSI & 布林寬度')
    )
    
    # 1. K線圖
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='K線',
            increasing_line_color='red',
            decreasing_line_color='green'
        ),
        row=1, col=1
    )
    
    # 移動平均線
    ma_colors = {'MA5': '#FF6B6B', 'MA10': '#4ECDC4', 'MA20': '#45B7D1', 'MA60': '#FFA07A'}
    for ma, color in ma_colors.items():
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[ma], 
                    name=ma,
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
    
    # 標示均線交叉點
    for cross_name, cross_info in crossovers.items():
        if cross_info['status'] == 'golden':
            days_ago = cross_info['days_ago']
            if days_ago < len(df):
                cross_date = df.index[-days_ago]
                cross_price = df.iloc[-days_ago]['Close']
                
                ma_pair = cross_name.replace('_cross_', ' x ')
                
                fig.add_trace(
                    go.Scatter(
                        x=[cross_date],
                        y=[cross_price],
                        mode='markers+text',
                        marker=dict(size=15, color='gold', symbol='star'),
                        text=[ma_pair],
                        textposition='top center',
                        name=f'{ma_pair} 黃金交叉',
                        showlegend=True
                    ),
                    row=1, col=1
                )
    
    # 布林通道
    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_upper'],
                name='BB上軌',
                line=dict(color='rgba(250, 128, 114, 0.5)', width=1, dash='dash')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_lower'],
                name='BB下軌',
                line=dict(color='rgba(250, 128, 114, 0.5)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(250, 128, 114, 0.1)'
            ),
            row=1, col=1
        )
    
    # 2. 成交量
    colors_volume = ['red' if close > open else 'green' 
                     for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='成交量',
            marker_color=colors_volume,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    if 'Volume_MA5' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Volume_MA5'],
                name='量MA5',
                line=dict(color='orange', width=1.5)
            ),
            row=2, col=1
        )
    
    # 3. MACD
    if 'MACD_hist' in df.columns:
        colors_macd = ['red' if val > 0 else 'green' for val in df['MACD_hist']]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_hist'],
                name='MACD柱',
                marker_color=colors_macd,
                opacity=0.7
            ),
            row=3, col=1
        )
    
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_signal'],
                name='Signal',
                line=dict(color='red', width=1.5)
            ),
            row=3, col=1
        )
    
    # 4. RSI 和布林寬度
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=4, col=1
        )
        
        # RSI 超買超賣線
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)
    
    # 布局設定
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    return fig

# ==================== 主程式 ====================

st.title("📈 台股進階分析系統 v3.0")
st.markdown("**均線交叉 × 布林收縮 × MACD轉強 - 多頭選股系統**")

# 建立分頁
tab1, tab2, tab3 = st.tabs(["🔍 單股分析", "📊 批次掃描", "📚 使用說明"])

# ==================== Tab 1: 單股分析 ====================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_code = st.text_input(
            "輸入股票代號",
            value="2330",
            help="輸入台股代號,例如: 2330 (台積電)"
        )
    
    with col2:
        period = st.selectbox(
            "時間範圍",
            options=["3mo", "6mo", "1y", "2y"],
            index=1,
            format_func=lambda x: {
                "3mo": "3個月",
                "6mo": "6個月", 
                "1y": "1年",
                "2y": "2年"
            }[x]
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        lookback_days = st.slider(
            "均線交叉回溯天數",
            min_value=3,
            max_value=20,
            value=10,
            help="檢查最近N天內的均線交叉"
        )
    
    with col4:
        bb_threshold = st.slider(
            "布林收縮閾值 (%)",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="布林寬度低於此值視為收縮"
        )
    
    analyze_button = st.button("🚀 開始分析", type="primary", use_container_width=True)
    
    if analyze_button:
        with st.spinner(f"正在分析 {stock_code}..."):
            df, stock = get_taiwan_stock_data(stock_code, period=period)
            
            if df is not None and not df.empty:
                # 計算技術指標
                df = calculate_technical_indicators(df)
                
                # 獲取股票資訊
                try:
                    info = stock.info
                    company_name = info.get('longName', stock_code)
                except:
                    company_name = stock_code
                
                st.success(f"✅ 成功抓取 {company_name} ({stock_code}) 的資料")
                
                # 偵測均線交叉
                crossovers = detect_ma_crossover(df, lookback=lookback_days)
                
                # 檢查多頭設定
                is_bullish, signals, score = check_bullish_setup(df)
                
                # 偵測布林收縮
                is_squeezed, current_width, avg_width = detect_bollinger_squeeze(
                    df, threshold=bb_threshold
                )
                
                # 顯示訊號卡片
                st.markdown("### 🎯 技術訊號")
                
                col_signal1, col_signal2, col_signal3 = st.columns(3)
                
                with col_signal1:
                    if is_bullish:
                        st.markdown(
                            f'<div class="signal-card signal-buy">'
                            f'<h3 style="margin:0; color:#00C853;">✅ 多頭設定</h3>'
                            f'<p style="margin:0.5rem 0 0 0; font-size:24px; font-weight:700;">評分: {score}分</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="signal-card signal-neutral">'
                            f'<h3 style="margin:0; color:#FFA726;">⚠️ 觀察中</h3>'
                            f'<p style="margin:0.5rem 0 0 0; font-size:24px; font-weight:700;">評分: {score}分</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                with col_signal2:
                    latest = df.iloc[-1]
                    if 'MACD_hist' in df.columns and pd.notna(latest['MACD_hist']):
                        macd_status = "紅柱" if latest['MACD_hist'] > 0 else "綠柱"
                        macd_color = "#00C853" if latest['MACD_hist'] > 0 else "#FF1744"
                        signal_class = "signal-buy" if latest['MACD_hist'] > 0 else "signal-sell"
                    else:
                        macd_status = "N/A"
                        macd_color = "#666"
                        signal_class = "signal-neutral"
                    
                    st.markdown(
                        f'<div class="signal-card {signal_class}">'
                        f'<h3 style="margin:0; color:{macd_color};">MACD</h3>'
                        f'<p style="margin:0.5rem 0 0 0; font-size:24px; font-weight:700;">{macd_status}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col_signal3:
                    if is_squeezed and current_width is not None:
                        st.markdown(
                            f'<div class="signal-card signal-buy">'
                            f'<h3 style="margin:0; color:#00C853;">布林收縮</h3>'
                            f'<p style="margin:0.5rem 0 0 0; font-size:24px; font-weight:700;">{current_width:.2f}%</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        width_text = f"{current_width:.2f}%" if current_width is not None else "N/A"
                        st.markdown(
                            f'<div class="signal-card signal-neutral">'
                            f'<h3 style="margin:0; color:#FFA726;">布林寬度</h3>'
                            f'<p style="margin:0.5rem 0 0 0; font-size:24px; font-weight:700;">{width_text}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                
                # 詳細訊號列表
                if signals:
                    st.markdown("### 📋 詳細訊號")
                    for signal in signals:
                        if '✅' in signal or '⭐' in signal:
                            st.success(signal)
                        elif '⚠️' in signal:
                            st.warning(signal)
                        else:
                            st.info(signal)
                
                # 均線交叉摘要
                st.markdown("### 📊 均線交叉摘要")
                
                col_cross1, col_cross2, col_cross3 = st.columns(3)
                
                for i, (cross_name, cross_info) in enumerate(crossovers.items()):
                    col = [col_cross1, col_cross2, col_cross3][i]
                    
                    with col:
                        ma_pair = cross_name.replace('_cross_', ' × ')
                        
                        if cross_info['status'] == 'golden':
                            st.success(f"**{ma_pair}**\n\n🟡 黃金交叉 ({cross_info['days_ago']}天前)")
                        elif cross_info['status'] == 'death':
                            st.error(f"**{ma_pair}**\n\n☠️ 死亡交叉 ({cross_info['days_ago']}天前)")
                        else:
                            st.info(f"**{ma_pair}**\n\n⚪ 無交叉")
                
                # 繪製圖表
                st.markdown("### 📈 技術圖表")
                fig = plot_candlestick_with_signals(df, stock_code, crossovers)
                st.plotly_chart(fig, use_container_width=True)
                
                # 最新數據
                st.markdown("### 📊 最新數據")
                
                latest = df.iloc[-1]
                
                col_data1, col_data2, col_data3, col_data4 = st.columns(4)
                
                with col_data1:
                    st.metric("收盤價", f"${latest['Close']:.2f}")
                    if 'MA20' in df.columns and pd.notna(latest['MA20']):
                        st.metric("MA20", f"${latest['MA20']:.2f}")
                
                with col_data2:
                    if 'RSI' in df.columns and pd.notna(latest['RSI']):
                        st.metric("RSI", f"{latest['RSI']:.2f}")
                    if 'ATR' in df.columns and pd.notna(latest['ATR']):
                        st.metric("ATR", f"{latest['ATR']:.2f}")
                
                with col_data3:
                    if 'K' in df.columns and pd.notna(latest['K']):
                        st.metric("KD-K", f"{latest['K']:.2f}")
                    if 'D' in df.columns and pd.notna(latest['D']):
                        st.metric("KD-D", f"{latest['D']:.2f}")
                
                with col_data4:
                    vol_ratio = latest['Volume'] / latest['Volume_MA5'] if pd.notna(latest['Volume_MA5']) else 0
                    st.metric("量能比", f"{vol_ratio:.2f}x")
                    st.metric("成交量", f"{latest['Volume']/1000:.0f}K")
            
            else:
                st.error(f"❌ 無法取得 {stock_code} 的資料,請確認股票代號是否正確")

# ==================== Tab 2: 批次掃描 ====================
with tab2:
    st.markdown("""
    ## 📊 批次選股掃描
    
    輸入多個股票代號(每行一個),系統會自動掃描符合多頭設定的股票。
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_list_input = st.text_area(
            "輸入股票代號列表 (每行一個)",
            value="2330\n2317\n2454\n3008\n2308",
            height=200,
            help="每行輸入一個股票代號"
        )
    
    with col2:
        st.markdown("### ⚙️ 掃描參數")
        
        lookback_days_batch = st.slider(
            "交叉回溯天數",
            min_value=5,
            max_value=20,
            value=10,
            key="batch_lookback"
        )
        
        bb_threshold_batch = st.slider(
            "布林收縮閾值 (%)",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            key="batch_bb"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        scan_button = st.button("🚀 開始掃描", type="primary", use_container_width=True)
    
    with col2:
        min_score = st.slider(
            "最低分數門檻",
            min_value=4,
            max_value=10,
            value=6,
            help="只顯示評分 >= 此值的股票"
        )
    
    if scan_button:
        stock_list = [s.strip() for s in stock_list_input.split('\n') if s.strip()]
        
        if not stock_list:
            st.warning("請輸入至少一個股票代號")
        else:
            st.markdown("---")
            st.subheader(f"掃描結果 (共 {len(stock_list)} 檔)")
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, code in enumerate(stock_list):
                status_text.text(f"正在分析 {code}... ({i+1}/{len(stock_list)})")
                progress_bar.progress((i + 1) / len(stock_list))
                
                try:
                    df, stock = get_taiwan_stock_data(code, period="6mo", interval="1d")
                    
                    if df is not None and not df.empty and len(df) >= 60:
                        df = calculate_technical_indicators(df)
                        is_bullish, signals, score = check_bullish_setup(df)
                        
                        if score >= min_score:
                            latest = df.iloc[-1]
                            
                            try:
                                info = stock.info
                                company_name = info.get('longName', code)
                            except:
                                company_name = code
                            
                            macd_status = 'N/A'
                            if 'MACD_hist' in df.columns and pd.notna(latest['MACD_hist']):
                                macd_status = '紅' if latest['MACD_hist'] > 0 else '綠'
                            
                            rsi_value = 'N/A'
                            if 'RSI' in df.columns and pd.notna(latest['RSI']):
                                rsi_value = f"{latest['RSI']:.1f}"
                            
                            results.append({
                                '代號': code,
                                '名稱': company_name,
                                '評分': score,
                                '收盤價': f"${latest['Close']:.2f}",
                                'RSI': rsi_value,
                                'MACD': macd_status,
                                '布林收縮': '是' if detect_bollinger_squeeze(df, threshold=bb_threshold_batch)[0] else '否',
                                '訊號數': len(signals)
                            })
                
                except Exception as e:
                    continue
            
            status_text.empty()
            progress_bar.empty()
            
            if results:
                st.success(f"✅ 找到 {len(results)} 檔符合條件的股票!")
                
                # 轉換成 DataFrame 並排序
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values('評分', ascending=False)
                
                # 顯示表格
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    hide_index=True
                )
                
                # 下載按鈕
                csv = df_results.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下載選股結果 (CSV)",
                    data=csv,
                    file_name=f"選股結果_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # 詳細訊號查看
                st.markdown("---")
                st.subheader("📋 詳細訊號")
                
                selected_stock = st.selectbox(
                    "選擇股票查看詳細訊號",
                    options=df_results['代號'].tolist()
                )
                
                if selected_stock:
                    df_stock, _ = get_taiwan_stock_data(selected_stock, period="6mo", interval="1d")
                    if df_stock is not None:
                        df_stock = calculate_technical_indicators(df_stock)
                        _, signals, _ = check_bullish_setup(df_stock)
                        
                        for signal in signals:
                            if '✅' in signal or '⭐' in signal:
                                st.success(signal)
                            elif '⚠️' in signal:
                                st.warning(signal)
                            else:
                                st.info(signal)
            
            else:
                st.warning("😢 沒有找到符合條件的股票,請嘗試降低分數門檻")

# ==================== Tab 3: 使用說明 ====================
with tab3:
    st.markdown("""
    ## 📚 v3.0 新功能說明
    
    ### 🎯 多頭選股邏輯
    
    系統會給每檔股票評分,滿分約 15-20 分,條件包括:
    
    #### 1. 均線交叉 (最高9分)
    - MA5 黃金交叉 MA10: +2分
    - MA10 黃金交叉 MA20: +3分 ⭐
    - MA20 黃金交叉 MA60: +4分 ⭐⭐
    
    #### 2. 布林通道收縮 (+3分)
    - 布林寬度 < 設定閾值(預設3%)
    - 代表即將突破,波動率壓縮
    
    #### 3. MACD 訊號 (最高5分)
    - MACD 紅柱: +2分
    - MACD 即將黃金交叉: +1分
    - MACD 剛黃金交叉: +3分 ⭐
    
    #### 4. 股價位置 (最高5分)
    - 股價站上 MA20: +2分
    - 多頭排列(價>MA5>MA10>MA20): +3分 ⭐
    
    #### 5. 成交量配合 (+1分)
    - 成交量 > 5日均量 × 1.2
    
    ---
    
    ### ⚠️ 風險提示
    
    1. **評分高不等於一定會漲** - 這是機率遊戲,必須配合停損
    2. **大盤走勢影響很大** - 大盤空頭時,個股很難獨強
    3. **基本面也很重要** - 技術面只看供需,還要搭配財報
    4. **假突破的存在** - 一定要設停損!
    
    ---
    
    **祝你選股成功!** 📈✨
    """)

# 頁腳
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem 0;'>
    <p>台股進階分析系統 v3.0 | 均線交叉 × 布林收縮 × MACD轉強</p>
    <p>⚠️ 投資有風險,請務必設定停損保護資金</p>
</div>
""", unsafe_allow_html=True)
