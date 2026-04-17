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
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

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
    df = df.copy()
    
    # 移動平均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # MACD - 使用 pandas_ta
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    
    # RSI - 使用 pandas_ta
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # KD (Stochastic) - 使用 pandas_ta
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
    df['K'] = stoch['STOCHk_9_3_3']
    df['D'] = stoch['STOCHd_9_3_3']
    
    # 布林通道 - 使用 pandas_ta
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df['BB_upper'] = bbands['BBU_20_2.0']
    df['BB_middle'] = bbands['BBM_20_2.0']
    df['BB_lower'] = bbands['BBL_20_2.0']
    
    # 布林寬度 (用於判斷收縮)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 100
    
    # ATR - 使用 pandas_ta
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # 成交量移動平均
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
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
    if len(df) < lookback:
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
    if is_squeezed:
        signals.append(f"✅ 布林通道收縮 (寬度:{current_width:.2f}%)")
        score += 3
    
    # 3. 檢查MACD
    if pd.notna(latest['MACD_hist']):
        if latest['MACD_hist'] > 0:
            signals.append(f"✅ MACD紅柱 ({latest['MACD_hist']:.3f})")
            score += 2
        elif latest['MACD_hist'] > -0.5 and latest['MACD'] > latest['MACD_signal']:
            signals.append(f"⚠️ MACD即將黃金交叉")
            score += 1
    
    # 檢查MACD是否剛黃金交叉
    if len(df) >= 2:
        prev = df.iloc[-2]
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
    colors_macd = ['red' if val > 0 else 'green' for val in df['MACD_hist']]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['MACD_hist'],
            name='MACD柱',
            marker_color=colors_macd
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='DIF', 
                   line=dict(color='blue', width=1.5)),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_signal'], name='DEA',
                   line=dict(color='red', width=1.5)),
        row=3, col=1
    )
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)
    
    # 4. RSI & 布林寬度
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                   line=dict(color='purple', width=2)),
        row=4, col=1
    )
    
    # 布林寬度 (用第二Y軸)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_width'], name='布林寬度(%)',
                   line=dict(color='orange', width=2, dash='dash'),
                   yaxis='y2'),
        row=4, col=1
    )
    
    # 添加 RSI 超買超賣線
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="超買", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green",
                  annotation_text="超賣", row=4, col=1)
    
    # 更新布局
    fig.update_layout(
        height=1200,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

# ==================== 主程式 ====================

st.title("📊 台股進階分析系統 v3.0")
st.markdown("**新功能**: 均線交叉偵測 | 布林收縮 | 多頭選股")

# 側邊欄
with st.sidebar:
    st.header("⚙️ 設定")
    
    stock_code = st.text_input(
        "股票代號", 
        value="2330",
        help="輸入台股代號"
    )
    
    period_options = {
        "3個月": "3mo",
        "6個月": "6mo",
        "1年": "1y",
        "2年": "2y"
    }
    
    period_label = st.selectbox(
        "時間範圍",
        options=list(period_options.keys()),
        index=2
    )
    period = period_options[period_label]
    
    interval_options = {
        "日線": "1d",
        "週線": "1wk"
    }
    
    interval_label = st.selectbox(
        "K線級別",
        options=list(interval_options.keys()),
        index=0
    )
    interval = interval_options[interval_label]
    
    st.markdown("---")
    
    st.subheader("🎯 訊號偵測設定")
    
    lookback_days = st.slider(
        "均線交叉回溯天數",
        min_value=3,
        max_value=20,
        value=10,
        help="檢查最近N天內的均線交叉"
    )
    
    bb_threshold = st.slider(
        "布林收縮閾值 (%)",
        min_value=1.5,
        max_value=5.0,
        value=3.0,
        step=0.5,
        help="布林寬度低於此值視為收縮"
    )
    
    st.markdown("---")
    
    if st.button("🔄 重新整理", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 主要內容
tab1, tab2, tab3 = st.tabs([
    "📈 技術分析 & 訊號", 
    "🔍 批次選股掃描",
    "📚 使用說明"
])

# ==================== Tab 1: 技術分析 ====================
with tab1:
    if stock_code:
        with st.spinner("正在分析..."):
            df, stock = get_taiwan_stock_data(stock_code, period, interval)
        
        if df is not None and not df.empty:
            df = calculate_technical_indicators(df)
            
            try:
                info = stock.info
                company_name = info.get('longName', stock_code)
            except:
                company_name = stock_code
            
            st.subheader(f"📊 {company_name} ({stock_code})")
            
            # 最新數據
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            change = latest['Close'] - prev['Close']
            change_pct = (change / prev['Close']) * 100
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "收盤價",
                    f"${latest['Close']:.2f}",
                    f"{change:+.2f} ({change_pct:+.2f}%)"
                )
            
            with col2:
                st.metric("RSI", f"{latest['RSI']:.1f}")
            
            with col3:
                st.metric("MACD柱", f"{latest['MACD_hist']:.3f}")
            
            with col4:
                st.metric("布林寬度", f"{latest['BB_width']:.2f}%")
            
            with col5:
                is_squeezed, _, _ = detect_bollinger_squeeze(df, threshold=bb_threshold)
                squeeze_status = "🔥 收縮中" if is_squeezed else "正常"
                st.metric("布林狀態", squeeze_status)
            
            st.markdown("---")
            
            # 多頭設定檢查
            st.subheader("🎯 多頭訊號評分")
            
            is_bullish, signals, score = check_bullish_setup(df)
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if is_bullish:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #00C853, #64DD17); 
                                padding: 2rem; border-radius: 10px; text-align: center;'>
                        <h1 style='color: white; margin: 0;'>⭐ {score} 分</h1>
                        <p style='color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
                            <strong>符合多頭設定!</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #FFA726, #FF6F00); 
                                padding: 2rem; border-radius: 10px; text-align: center;'>
                        <h1 style='color: white; margin: 0;'>{score} 分</h1>
                        <p style='color: white; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
                            尚未達標
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 訊號明細")
                if signals:
                    for signal in signals:
                        if '✅' in signal or '⭐' in signal:
                            st.success(signal)
                        elif '⚠️' in signal:
                            st.warning(signal)
                        else:
                            st.info(signal)
                else:
                    st.info("目前無明顯訊號")
            
            st.markdown("---")
            
            # 均線交叉偵測
            st.subheader("🔄 均線交叉分析")
            
            crossovers = detect_ma_crossover(df, lookback=lookback_days)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cross_info = crossovers['MA5_cross_MA10']
                if cross_info['status'] == 'golden':
                    st.success(f"⭐ MA5 黃金交叉 MA10\n\n{cross_info['days_ago']}天前")
                elif cross_info['status'] == 'death':
                    st.error(f"💀 MA5 死亡交叉 MA10\n\n{cross_info['days_ago']}天前")
                else:
                    st.info("MA5 x MA10\n\n近期無交叉")
            
            with col2:
                cross_info = crossovers['MA10_cross_MA20']
                if cross_info['status'] == 'golden':
                    st.success(f"⭐ MA10 黃金交叉 MA20\n\n{cross_info['days_ago']}天前")
                elif cross_info['status'] == 'death':
                    st.error(f"💀 MA10 死亡交叉 MA20\n\n{cross_info['days_ago']}天前")
                else:
                    st.info("MA10 x MA20\n\n近期無交叉")
            
            with col3:
                cross_info = crossovers['MA20_cross_MA60']
                if cross_info['status'] == 'golden':
                    st.success(f"⭐ MA20 黃金交叉 MA60\n\n{cross_info['days_ago']}天前")
                elif cross_info['status'] == 'death':
                    st.error(f"💀 MA20 死亡交叉 MA60\n\n{cross_info['days_ago']}天前")
                else:
                    st.info("MA20 x MA60\n\n近期無交叉")
            
            st.markdown("---")
            
            # 布林收縮分析
            st.subheader("📏 布林通道分析")
            
            is_squeezed, current_width, avg_width = detect_bollinger_squeeze(
                df, threshold=bb_threshold
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if is_squeezed:
                    st.success(f"""
                    ### 🔥 布林通道收縮中!
                    
                    - **目前寬度**: {current_width:.2f}%
                    - **平均寬度**: {avg_width:.2f}%
                    - **狀態**: 準備突破
                    
                    💡 **建議**: 關注突破方向,配合其他指標確認
                    """)
                else:
                    st.info(f"""
                    ### 📊 布林通道正常
                    
                    - **目前寬度**: {current_width:.2f}%
                    - **平均寬度**: {avg_width:.2f}%
                    - **狀態**: 尚未收縮
                    """)
            
            with col2:
                # 布林寬度走勢圖
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(
                    x=df.index[-60:],
                    y=df['BB_width'].iloc[-60:],
                    name='布林寬度',
                    line=dict(color='orange', width=2)
                ))
                fig_bb.add_hline(
                    y=bb_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"收縮閾值 ({bb_threshold}%)"
                )
                fig_bb.update_layout(
                    title="近60日布林寬度",
                    yaxis_title="寬度 (%)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_bb, use_container_width=True)
            
            st.markdown("---")
            
            # K線圖表
            st.subheader("📊 K線圖表")
            fig = plot_candlestick_with_signals(df, stock_code, crossovers)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"❌ 無法取得 {stock_code} 的數據")

# ==================== Tab 2: 批次選股 ====================
with tab2:
    st.subheader("🔍 批次選股掃描器")
    
    st.info("""
    **功能說明**: 輸入多個股票代號,系統會批次掃描並找出符合多頭設定的股票
    
    **篩選條件**:
    - ✅ 均線黃金交叉
    - ✅ 布林通道收縮
    - ✅ MACD轉紅
    - ✅ 多頭排列
    - ✅ 成交量配合
    """)
    
    # 輸入股票清單
    stock_list_input = st.text_area(
        "輸入股票代號 (每行一個)",
        value="2330\n2317\n2454\n3008\n2382",
        height=150,
        help="每行輸入一個股票代號"
    )
    
    col1, col2 = st.columns([1, 3])
    
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
                            
                            results.append({
                                '代號': code,
                                '名稱': company_name,
                                '評分': score,
                                '收盤價': f"${latest['Close']:.2f}",
                                'RSI': f"{latest['RSI']:.1f}",
                                'MACD': '紅' if latest['MACD_hist'] > 0 else '綠',
                                '布林收縮': '是' if detect_bollinger_squeeze(df, threshold=bb_threshold)[0] else '否',
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
    
    ### 📈 使用策略建議
    
    #### 策略 A: 保守型 (建議評分 ≥ 8分)
    ```
    1. 批次掃描,篩選評分 ≥ 8分
    2. 重點關注有「MA10黃金交叉MA20」的股票
    3. 確認布林收縮 + MACD紅柱
    4. 等股價站穩MA20後進場
    5. 停損: 跌破MA20
    ```
    
    #### 策略 B: 積極型 (建議評分 ≥ 6分)
    ```
    1. 批次掃描,篩選評分 ≥ 6分
    2. 關注「布林收縮 + MACD即將黃金交叉」
    3. 搶先布局,等待突破
    4. 突破後加碼
    5. 停損: 跌破MA10
    ```
    
    #### 策略 C: 波段操作
    ```
    1. 尋找「MA20黃金交叉MA60」(中長期多頭)
    2. 等待回檔至MA20
    3. 出現「MA5黃金交叉MA10」時進場
    4. 目標: 前高或MA60乖離 +20%
    5. 停損: 跌破MA60
    ```
    
    ---
    
    ### 🎓 指標解讀
    
    #### 均線黃金交叉的意義
    - **MA5 x MA10**: 短線轉強訊號
    - **MA10 x MA20**: 中期趨勢確立 (重要!)
    - **MA20 x MA60**: 長期多頭啟動 (最重要!)
    
    #### 布林收縮的意義
    - 代表盤整,波動率降低
    - 通常在突破前發生
    - 收縮後突破,力道往往很強
    - **方向未定,需搭配其他指標**
    
    #### MACD 轉紅的意義
    - 短期動能轉強
    - 柱狀圖由綠轉紅是關鍵
    - 配合黃金交叉最佳
    
    ---
    
    ### ⚠️ 風險提示
    
    1. **評分高不等於一定會漲**
       - 這是機率遊戲
       - 必須配合停損
    
    2. **大盤走勢影響很大**
       - 大盤空頭時,個股很難獨強
       - 建議先確認大盤趨勢
    
    3. **基本面也很重要**
       - 技術面只看供需
       - 還要搭配財報、產業趨勢
    
    4. **假突破的存在**
       - 布林收縮後可能向下突破
       - 黃金交叉後可能再交叉回來
       - **一定要設停損!**
    
    ---
    
    ### 💡 實戰技巧
    
    #### 技巧 1: 建立觀察名單
    ```
    1. 每週日用批次掃描找出高分股票
    2. 評分 ≥ 6分放入觀察名單
    3. 評分 ≥ 8分重點追蹤
    4. 每天檢查訊號變化
    ```
    
    #### 技巧 2: 分批進場
    ```
    1. 評分達標後不要 all-in
    2. 先買 1/3 部位試水溫
    3. 突破確認後加碼 1/3
    4. 趨勢確立後最後 1/3
    ```
    
    #### 技巧 3: 設定提醒
    ```
    1. 記錄每檔股票的關鍵價位
    2. MA20、MA60 的價格
    3. 前高、前低
    4. 突破/跌破時要果斷行動
    ```
    
    #### 技巧 4: 定期檢視
    ```
    1. 持股的評分可能會降低
    2. 如果評分 < 4分要考慮減碼
    3. 死亡交叉要果斷停損
    4. 不要死抱跌破關鍵均線的股票
    ```
    
    ---
    
    ### 📊 參數調整建議
    
    #### 均線交叉回溯天數 (預設10天)
    - 增加: 只抓較舊的交叉(避免剛交叉就抓到)
    - 減少: 只抓最近的交叉(更即時)
    
    #### 布林收縮閾值 (預設3%)
    - 降低(2%): 更嚴格,只抓極度收縮
    - 提高(4%): 較寬鬆,提早發現
    
    #### 最低評分門檻 (預設6分)
    - 提高(8分): 更嚴格篩選,減少假訊號
    - 降低(4分): 更多機會,但品質參差
    
    ---
    
    ### 🚀 進階功能(未來開發)
    
    - [ ] 自動每日掃描並發送通知
    - [ ] 回測功能驗證策略
    - [ ] 整合基本面數據
    - [ ] AI 預測突破方向
    - [ ] 自動追蹤持股訊號變化
    
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
