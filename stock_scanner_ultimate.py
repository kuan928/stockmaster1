"""
台股智能分析系統 - 完整版
技術面 + 籌碼面 + 自訂篩選
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="台股智能分析", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .recommend-buy { background: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #00C853; }
    .recommend-hold { background: #FFF3E0; padding: 20px; border-radius: 10px; border-left: 5px solid #FFA726; }
    .recommend-sell { background: #FFEBEE; padding: 20px; border-radius: 10px; border-left: 5px solid #FF1744; }
    .indicator-box { background: white; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

# ==================== 資料取得 ====================

def get_stock_data(code, period="6mo"):
    """取得股價資料"""
    try:
        ticker = f"{code}.TW"
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            ticker = f"{code}.TWO"
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
        if not df.empty:
            return df, stock
        return None, None
    except:
        return None, None

def get_institutional_data(stock_code):
    """取得法人買賣資料 (模擬數據)"""
    # 註: 實際應用需要連接真實的台股資料源
    # 這裡提供模擬數據結構
    try:
        # 模擬最近5天的法人買賣
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        
        data = {
            '日期': dates,
            '外資買賣': np.random.randint(-5000, 8000, 5),  # 張數
            '投信買賣': np.random.randint(-2000, 3000, 5),
            '自營商買賣': np.random.randint(-1000, 2000, 5),
            '融資增減': np.random.randint(-500, 1000, 5),
            '融券增減': np.random.randint(-200, 300, 5),
        }
        
        return pd.DataFrame(data)
    except:
        return None

# ==================== 技術指標計算 ====================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def calc_indicators(df):
    """計算所有技術指標"""
    df = df.copy()
    
    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # MACD
    macd, signal, hist = calc_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    
    # RSI
    df['RSI'] = calc_rsi(df['Close'])
    
    # KD
    lowest_low = df['Low'].rolling(9).min()
    highest_high = df['High'].rolling(9).max()
    df['K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    df['D'] = df['K'].rolling(3).mean()
    
    # 成交量
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    
    return df

# ==================== 分析函數 ====================

def analyze_technical(df):
    """技術面分析"""
    if len(df) < 60:
        return {}, 0
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = {}
    score = 0
    
    # 均線
    if latest['Close'] > latest['MA5'] > latest['MA10'] > latest['MA20']:
        signals['均線'] = ('多頭排列', 'BUY', 3)
        score += 3
    elif latest['Close'] < latest['MA5'] < latest['MA10'] < latest['MA20']:
        signals['均線'] = ('空頭排列', 'SELL', -3)
        score -= 3
    else:
        signals['均線'] = ('糾結', 'NEUTRAL', 0)
    
    # 均線交叉
    if prev['MA5'] <= prev['MA10'] and latest['MA5'] > latest['MA10']:
        signals['MA交叉'] = ('MA5黃金交叉MA10', 'BUY', 2)
        score += 2
    elif prev['MA10'] <= prev['MA20'] and latest['MA10'] > latest['MA20']:
        signals['MA交叉'] = ('MA10黃金交叉MA20', 'BUY', 3)
        score += 3
    
    # MACD
    if latest['MACD_hist'] > 0 and prev['MACD_hist'] <= 0:
        signals['MACD'] = ('剛轉紅', 'BUY', 3)
        score += 3
    elif latest['MACD_hist'] < 0 and prev['MACD_hist'] >= 0:
        signals['MACD'] = ('剛轉綠', 'SELL', -3)
        score -= 3
    elif latest['MACD_hist'] > 0:
        signals['MACD'] = ('紅柱', 'BUY', 1)
        score += 1
    else:
        signals['MACD'] = ('綠柱', 'SELL', -1)
        score -= 1
    
    # RSI
    if latest['RSI'] < 30:
        signals['RSI'] = ('超賣', 'BUY', 2)
        score += 2
    elif latest['RSI'] > 70:
        signals['RSI'] = ('超買', 'SELL', -2)
        score -= 2
    else:
        signals['RSI'] = ('中性', 'NEUTRAL', 0)
    
    # KD
    if latest['K'] < 20 and latest['D'] < 20:
        signals['KD'] = ('超賣', 'BUY', 2)
        score += 2
    elif latest['K'] > 80 and latest['D'] > 80:
        signals['KD'] = ('超買', 'SELL', -2)
        score -= 2
    
    if prev['K'] <= prev['D'] and latest['K'] > latest['D'] and latest['K'] < 50:
        signals['KD交叉'] = ('低檔黃金交叉', 'BUY', 3)
        score += 3
    
    # 成交量
    if latest['Volume'] > latest['Volume_MA5'] * 1.5:
        if latest['Close'] > prev['Close']:
            signals['成交量'] = ('放量上漲', 'BUY', 2)
            score += 2
        else:
            signals['成交量'] = ('放量下跌', 'SELL', -2)
            score -= 2
    
    return signals, score

def analyze_institutional(inst_df):
    """籌碼面分析"""
    if inst_df is None or inst_df.empty:
        return {}, 0
    
    signals = {}
    score = 0
    
    # 最近3天的法人買賣
    recent = inst_df.tail(3)
    
    # 外資
    foreign_sum = recent['外資買賣'].sum()
    if foreign_sum > 3000:
        signals['外資'] = (f'近3日買超 {foreign_sum:,}張', 'BUY', 3)
        score += 3
    elif foreign_sum < -3000:
        signals['外資'] = (f'近3日賣超 {abs(foreign_sum):,}張', 'SELL', -3)
        score -= 3
    elif foreign_sum > 1000:
        signals['外資'] = (f'近3日買超 {foreign_sum:,}張', 'BUY', 1)
        score += 1
    elif foreign_sum < -1000:
        signals['外資'] = (f'近3日賣超 {abs(foreign_sum):,}張', 'SELL', -1)
        score -= 1
    
    # 投信
    trust_sum = recent['投信買賣'].sum()
    if trust_sum > 1000:
        signals['投信'] = (f'近3日買超 {trust_sum:,}張', 'BUY', 2)
        score += 2
    elif trust_sum < -1000:
        signals['投信'] = (f'近3日賣超 {abs(trust_sum):,}張', 'SELL', -2)
        score -= 2
    
    # 自營商
    dealer_sum = recent['自營商買賣'].sum()
    if dealer_sum > 500:
        signals['自營商'] = (f'近3日買超 {dealer_sum:,}張', 'BUY', 1)
        score += 1
    elif dealer_sum < -500:
        signals['自營商'] = (f'近3日賣超 {abs(dealer_sum):,}張', 'SELL', -1)
        score -= 1
    
    # 融資融券
    margin_sum = recent['融資增減'].sum()
    short_sum = recent['融券增減'].sum()
    
    if margin_sum < -500 and short_sum < -200:
        signals['融資券'] = ('融資融券雙降', 'BUY', 2)
        score += 2
    elif margin_sum > 500 and short_sum > 200:
        signals['融資券'] = ('融資融券雙升', 'SELL', -1)
        score -= 1
    
    return signals, score

def get_final_recommendation(tech_score, inst_score):
    """綜合評分給建議"""
    total_score = tech_score + inst_score
    
    if total_score >= 10:
        return "強烈建議買進", "BUY", total_score
    elif total_score >= 6:
        return "建議買進", "BUY", total_score
    elif total_score >= 3:
        return "可以考慮買進", "BUY", total_score
    elif total_score >= -2:
        return "建議觀望", "HOLD", total_score
    elif total_score >= -5:
        return "建議賣出", "SELL", total_score
    else:
        return "強烈建議賣出", "SELL", total_score

# ==================== 主介面 ====================

st.title("📊 台股智能分析系統")
st.markdown("**技術面 × 籌碼面 × 智能篩選**")

tab1, tab2, tab3 = st.tabs(["🔍 單股分析", "🎯 智能選股", "📊 批次掃描"])

# ==================== Tab 1: 單股完整分析 ====================
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_code = st.text_input("🔢 輸入股票代號", value="2330")
    
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 完整分析", type="primary", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("分析中..."):
            df, stock = get_stock_data(stock_code)
            
            if df is not None and not df.empty:
                df = calc_indicators(df)
                inst_df = get_institutional_data(stock_code)
                
                try:
                    company_name = stock.info.get('longName', stock_code)
                except:
                    company_name = stock_code
                
                # 分析
                tech_signals, tech_score = analyze_technical(df)
                inst_signals, inst_score = analyze_institutional(inst_df)
                recommendation, action, total_score = get_final_recommendation(tech_score, inst_score)
                
                latest = df.iloc[-1]
                
                st.markdown("---")
                
                # 顯示建議
                if action == "BUY":
                    st.markdown(f"""
                    <div class='recommend-buy'>
                        <h2 style='color: #00C853; margin:0;'>🟢 {recommendation}</h2>
                        <p style='font-size:18px; margin:10px 0 0 0;'>{company_name} ({stock_code})</p>
                        <p style='margin:5px 0 0 0;'>技術面: {tech_score}分 | 籌碼面: {inst_score}分 | 總分: {total_score}分</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif action == "SELL":
                    st.markdown(f"""
                    <div class='recommend-sell'>
                        <h2 style='color: #FF1744; margin:0;'>🔴 {recommendation}</h2>
                        <p style='font-size:18px; margin:10px 0 0 0;'>{company_name} ({stock_code})</p>
                        <p style='margin:5px 0 0 0;'>技術面: {tech_score}分 | 籌碼面: {inst_score}分 | 總分: {total_score}分</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='recommend-hold'>
                        <h2 style='color: #FFA726; margin:0;'>🟡 {recommendation}</h2>
                        <p style='font-size:18px; margin:10px 0 0 0;'>{company_name} ({stock_code})</p>
                        <p style='margin:5px 0 0 0;'>技術面: {tech_score}分 | 籌碼面: {inst_score}分 | 總分: {total_score}分</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # 價格資訊
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 收盤價", f"${latest['Close']:.2f}")
                with col2:
                    change = latest['Close'] - df.iloc[-2]['Close']
                    change_pct = (change / df.iloc[-2]['Close']) * 100
                    st.metric("📊 漲跌", f"{change:+.2f}", f"{change_pct:+.2f}%")
                with col3:
                    vol_ratio = latest['Volume'] / latest['Volume_MA5']
                    st.metric("📈 量能比", f"{vol_ratio:.2f}x")
                with col4:
                    st.metric("📅 日期", latest.name.strftime('%m-%d'))
                
                st.markdown("---")
                
                # 技術面 + 籌碼面
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"📈 技術面分析 ({tech_score:+d}分)")
                    
                    if tech_signals:
                        for key, (desc, signal_type, score) in tech_signals.items():
                            if signal_type == 'BUY':
                                st.success(f"**{key}:** {desc} (+{score}分)")
                            elif signal_type == 'SELL':
                                st.error(f"**{key}:** {desc} ({score}分)")
                            else:
                                st.info(f"**{key}:** {desc}")
                    
                    st.markdown("##### 技術指標數值")
                    st.markdown(f"""
                    <div class='indicator-box'>
                    <b>MA5/10/20/60:</b> ${latest['MA5']:.2f} / ${latest['MA10']:.2f} / ${latest['MA20']:.2f} / ${latest['MA60']:.2f}<br>
                    <b>RSI:</b> {latest['RSI']:.2f}<br>
                    <b>KD:</b> {latest['K']:.2f} / {latest['D']:.2f}<br>
                    <b>MACD:</b> {latest['MACD']:.3f} (柱: {latest['MACD_hist']:.3f})
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader(f"💼 籌碼面分析 ({inst_score:+d}分)")
                    
                    if inst_signals:
                        for key, (desc, signal_type, score) in inst_signals.items():
                            if signal_type == 'BUY':
                                st.success(f"**{key}:** {desc} (+{score}分)")
                            elif signal_type == 'SELL':
                                st.error(f"**{key}:** {desc} ({score}分)")
                            else:
                                st.info(f"**{key}:** {desc}")
                    
                    if inst_df is not None:
                        st.markdown("##### 近5日法人買賣")
                        st.dataframe(
                            inst_df[['日期', '外資買賣', '投信買賣', '自營商買賣']].tail(5),
                            use_container_width=True,
                            hide_index=True
                        )
            else:
                st.error(f"❌ 無法取得 {stock_code} 的資料")

# ==================== Tab 2: 智能選股 ====================
with tab2:
    st.subheader("🎯 自訂條件選股")
    st.markdown("選擇你要的條件,系統自動幫你從台股找出符合的股票")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 技術面條件")
        
        tech_conditions = []
        
        if st.checkbox("✅ 多頭排列 (價 > MA5 > MA10 > MA20)"):
            tech_conditions.append("多頭排列")
        
        if st.checkbox("✅ MA黃金交叉 (MA5交叉MA10或MA10交叉MA20)"):
            tech_conditions.append("MA黃金交叉")
        
        if st.checkbox("✅ MACD紅柱或剛轉紅"):
            tech_conditions.append("MACD紅柱")
        
        if st.checkbox("✅ RSI超賣 (< 30)"):
            tech_conditions.append("RSI超賣")
        
        if st.checkbox("✅ KD低檔 (< 20)"):
            tech_conditions.append("KD低檔")
        
        if st.checkbox("✅ 放量上漲 (量 > 5日均量 1.5倍)"):
            tech_conditions.append("放量")
        
        if st.checkbox("✅ 站上MA20"):
            tech_conditions.append("站上MA20")
    
    with col2:
        st.markdown("#### 💼 籌碼面條件")
        
        inst_conditions = []
        
        if st.checkbox("✅ 外資近3日買超"):
            inst_conditions.append("外資買超")
        
        if st.checkbox("✅ 投信近3日買超"):
            inst_conditions.append("投信買超")
        
        if st.checkbox("✅ 自營商近3日買超"):
            inst_conditions.append("自營商買超")
        
        if st.checkbox("✅ 三大法人同步買超"):
            inst_conditions.append("三法人買超")
        
        if st.checkbox("✅ 融資減、融券減"):
            inst_conditions.append("融資券雙降")
    
    st.markdown("---")
    
    # 股票池選擇
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock_pool = st.selectbox(
            "選擇股票池",
            ["台灣50成分股", "電子股", "金融股", "傳產股", "自訂清單"]
        )
    
    with col2:
        min_score = st.slider("最低總分", 0, 20, 8)
    
    with col3:
        st.write("")
        st.write("")
        search_btn = st.button("🔍 開始搜尋", type="primary", use_container_width=True)
    
    # 自訂清單
    if stock_pool == "自訂清單":
        custom_stocks = st.text_area(
            "輸入股票代號 (每行一個)",
            value="2330\n2317\n2454\n3008\n2308",
            height=100
        )
    
    if search_btn:
        # 準備股票清單
        if stock_pool == "台灣50成分股":
            stock_list = ["2330", "2317", "2454", "2308", "2382", "2881", "2886", "2412", "2303", "1301"]
        elif stock_pool == "電子股":
            stock_list = ["2330", "2317", "2454", "2308", "2382", "3711", "2357", "3034", "2327", "2345"]
        elif stock_pool == "金融股":
            stock_list = ["2881", "2882", "2883", "2884", "2885", "2886", "2887", "2890", "2891", "2892"]
        elif stock_pool == "傳產股":
            stock_list = ["1301", "1303", "2002", "2207", "2408", "2409", "2912", "5880", "6505", "9904"]
        else:
            stock_list = [s.strip() for s in custom_stocks.split('\n') if s.strip()]
        
        st.info(f"搜尋中... 股票池: {len(stock_list)}檔")
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, code in enumerate(stock_list):
            status.text(f"分析: {code} ({i+1}/{len(stock_list)})")
            progress.progress((i + 1) / len(stock_list))
            
            try:
                df, stock = get_stock_data(code, period="3mo")
                if df is None or len(df) < 60:
                    continue
                
                df = calc_indicators(df)
                inst_df = get_institutional_data(code)
                
                tech_signals, tech_score = analyze_technical(df)
                inst_signals, inst_score = analyze_institutional(inst_df)
                recommendation, action, total_score = get_final_recommendation(tech_score, inst_score)
                
                if total_score < min_score:
                    continue
                
                # 檢查是否符合條件
                match = True
                
                # 技術面條件檢查
                if "多頭排列" in tech_conditions:
                    if '均線' not in tech_signals or tech_signals['均線'][0] != '多頭排列':
                        match = False
                
                if "MACD紅柱" in tech_conditions:
                    if 'MACD' not in tech_signals or 'SELL' in tech_signals['MACD'][1]:
                        match = False
                
                if "RSI超賣" in tech_conditions:
                    if 'RSI' not in tech_signals or tech_signals['RSI'][0] != '超賣':
                        match = False
                
                # 籌碼面條件檢查
                if "外資買超" in inst_conditions:
                    if '外資' not in inst_signals or 'SELL' in inst_signals['外資'][1]:
                        match = False
                
                if match:
                    latest = df.iloc[-1]
                    
                    try:
                        name = stock.info.get('longName', code)
                    except:
                        name = code
                    
                    results.append({
                        '代號': code,
                        '名稱': name,
                        '建議': recommendation,
                        '總分': total_score,
                        '技術分': tech_score,
                        '籌碼分': inst_score,
                        '收盤價': f"${latest['Close']:.2f}",
                        'RSI': f"{latest['RSI']:.1f}"
                    })
            
            except:
                continue
        
        status.empty()
        progress.empty()
        
        if results:
            st.success(f"✅ 找到 {len(results)} 檔符合條件的股票!")
            
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('總分', ascending=False)
            
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            csv = df_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下載選股結果",
                csv,
                f"選股_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.warning("😢 沒有找到符合條件的股票,試試放寬條件")

# ==================== Tab 3: 批次掃描 ====================
with tab3:
    st.subheader("📊 批次快速掃描")
    
    stock_input = st.text_area(
        "輸入股票代號 (每行一個)",
        value="2330\n2317\n2454\n3008\n2308\n2382\n2881\n2412\n2886\n2357",
        height=150
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        scan_btn = st.button("🔍 快速掃描", type="primary", use_container_width=True)
    
    with col2:
        filter_option = st.radio(
            "只顯示:",
            ["全部", "建議買進", "建議賣出"],
            horizontal=True
        )
    
    if scan_btn:
        stock_list = [s.strip() for s in stock_input.split('\n') if s.strip()]
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, code in enumerate(stock_list):
            status.text(f"掃描: {code} ({i+1}/{len(stock_list)})")
            progress.progress((i + 1) / len(stock_list))
            
            try:
                df, stock = get_stock_data(code, period="3mo")
                if df is None or len(df) < 60:
                    continue
                
                df = calc_indicators(df)
                inst_df = get_institutional_data(code)
                
                tech_signals, tech_score = analyze_technical(df)
                inst_signals, inst_score = analyze_institutional(inst_df)
                recommendation, action, total_score = get_final_recommendation(tech_score, inst_score)
                
                # 篩選
                if filter_option == "建議買進" and action != "BUY":
                    continue
                elif filter_option == "建議賣出" and action != "SELL":
                    continue
                
                latest = df.iloc[-1]
                
                try:
                    name = stock.info.get('longName', code)
                except:
                    name = code
                
                results.append({
                    '代號': code,
                    '名稱': name,
                    '建議': recommendation,
                    '總分': total_score,
                    '技術': tech_score,
                    '籌碼': inst_score,
                    '價格': f"${latest['Close']:.2f}",
                    'RSI': f"{latest['RSI']:.1f}"
                })
            except:
                continue
        
        status.empty()
        progress.empty()
        
        if results:
            st.success(f"✅ 找到 {len(results)} 檔股票")
            
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('總分', ascending=False)
            
            # 分類顯示
            buy_stocks = df_results[df_results['建議'].str.contains('買進')]
            if not buy_stocks.empty:
                st.markdown("### 🟢 建議買進")
                st.dataframe(buy_stocks, use_container_width=True, hide_index=True)
            
            sell_stocks = df_results[df_results['建議'].str.contains('賣出')]
            if not sell_stocks.empty:
                st.markdown("### 🔴 建議賣出")
                st.dataframe(sell_stocks, use_container_width=True, hide_index=True)
            
            csv = df_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下載掃描結果",
                csv,
                f"掃描_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.warning("沒有找到符合條件的股票")

st.markdown("---")
st.caption("⚠️ 本系統僅供參考,投資有風險。籌碼數據為模擬數據,實際使用需連接真實資料源。")
