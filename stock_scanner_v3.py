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
    .valuation-cheap { background: #C8E6C9; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; margin: 10px 0; }
    .valuation-fair { background: #FFF9C4; padding: 15px; border-radius: 8px; border-left: 5px solid #FFC107; margin: 10px 0; }
    .valuation-expensive { background: #FFCDD2; padding: 15px; border-radius: 8px; border-left: 5px solid #F44336; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

# ==================== 資料取得 ====================

def _clean_history(df):
    """丟掉今天尚未收盤的 NaN Close 列,保留完整交易日。"""
    if df is None or df.empty:
        return df
    df = df.dropna(subset=['Close'])
    return df


def get_stock_data(code, period="6mo"):
    """取得股價資料，並回傳實際使用的 ticker 代號 (.TW 或 .TWO)。"""
    try:
        code = str(code).strip().upper()
        if not code:
            return None, None, None

        # 已含副檔名 (例: 2330.TW, AAPL.US) 或純英文 (美股)
        if '.' in code or not code.replace('-', '').isdigit():
            stock = yf.Ticker(code)
            df = _clean_history(stock.history(period=period))
            if df is not None and not df.empty:
                return df, stock, code
            return None, None, None

        # 台股: 先試 .TW 再試 .TWO
        ticker = f"{code}.TW"
        stock = yf.Ticker(ticker)
        df = _clean_history(stock.history(period=period))
        if df is None or df.empty:
            ticker = f"{code}.TWO"
            stock = yf.Ticker(ticker)
            df = _clean_history(stock.history(period=period))
        if df is not None and not df.empty:
            return df, stock, ticker
        return None, None, None
    except Exception:
        return None, None, None

def _parse_int(v):
    """字串轉整數 (處理 '1,234,567'、'-'、空白)。"""
    if v is None:
        return 0
    try:
        s = str(v).replace(',', '').replace(' ', '').strip()
        if s in ('', '-', '--'):
            return 0
        return int(float(s))
    except Exception:
        return 0


def _roc_date(dt):
    """datetime → ROC 年/月/日 (例: 2026-04-17 → 115/04/17)"""
    return f"{dt.year - 1911}/{dt.month:02d}/{dt.day:02d}"


def _recent_trading_dates(n=5):
    """回傳最近 n 個交易日 (跳過週末;若為當天且收盤前,跳過當天)"""
    dates = []
    now = datetime.now()
    d = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # TWSE/TPEX 資料約 15:00 後才公布,16:00 前保守跳過當日
    if now.hour < 16:
        d = d - timedelta(days=1)
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d = d - timedelta(days=1)
    return dates


# 最近一次法人資料抓取錯誤 (供 UI 顯示診斷)
_INSTI_LAST_ERROR = {"value": None}

_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}


def _df_from_any(j):
    """TWSE/TPEX 新舊版回傳格式統一解析。"""
    # 新版: 頂層 tables: [{title, fields, data}, ...]
    tables = j.get('tables') or []
    candidates = []
    for t in tables:
        fields = t.get('fields') or []
        data = t.get('data') or []
        if fields and data and len(data[0]) == len(fields):
            candidates.append((t.get('title', ''), fields, data))
    # 有多個 table 時,挑欄位數最多者 (通常是主表)
    if candidates:
        _, fields, data = max(candidates, key=lambda c: len(c[1]))
        df = pd.DataFrame(data, columns=[str(c).strip() for c in fields])
        return df
    # 舊版: 頂層 fields + data
    if j.get('fields') and j.get('data'):
        fields = j['fields']
        data = j['data']
        if data and len(data[0]) == len(fields):
            return pd.DataFrame(data, columns=[str(c).strip() for c in fields])
    # 更舊版: aaData
    if j.get('aaData'):
        return pd.DataFrame(j['aaData'])
    return None


@st.cache_data(ttl=14400, show_spinner=False)
def _fetch_twse_insti_day(date_ymd):
    """抓 TWSE 某日三大法人 (YYYYMMDD)。回傳 DataFrame 或 None。"""
    try:
        url = "https://www.twse.com.tw/rwd/zh/fund/T86"
        params = {"date": date_ymd, "selectType": "ALL", "response": "json"}
        headers = dict(_HTTP_HEADERS)
        headers["Referer"] = "https://www.twse.com.tw/"
        r = requests.get(url, params=params, timeout=8, headers=headers)
        if r.status_code != 200:
            _INSTI_LAST_ERROR["value"] = f"TWSE HTTP {r.status_code}"
            return None
        j = r.json()
        # stat 可能不存在於新版回傳;優先以 tables/fields 判斷是否有資料
        df = _df_from_any(j)
        if df is None or df.empty:
            _INSTI_LAST_ERROR["value"] = (
                f"TWSE 無資料 stat={j.get('stat','?')} date={date_ymd} "
                f"keys={list(j.keys())[:5]}"
            )
            return None
        return df
    except requests.exceptions.Timeout:
        _INSTI_LAST_ERROR["value"] = "TWSE 連線逾時 (8秒)"
        return None
    except Exception as e:
        _INSTI_LAST_ERROR["value"] = f"TWSE 錯誤: {type(e).__name__}: {e}"
        return None


@st.cache_data(ttl=14400, show_spinner=False)
def _fetch_tpex_insti_day(date_roc):
    """抓 TPEX 某日三大法人 (ROC: YYY/MM/DD)。回傳 DataFrame 或 None。"""
    try:
        url = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
        params = {"type": "Daily", "sect": "AL", "date": date_roc,
                  "id": "", "response": "json"}
        headers = dict(_HTTP_HEADERS)
        headers["Referer"] = "https://www.tpex.org.tw/"
        r = requests.get(url, params=params, timeout=8, headers=headers)
        if r.status_code != 200:
            _INSTI_LAST_ERROR["value"] = f"TPEX HTTP {r.status_code}"
            return None
        j = r.json()
        df = _df_from_any(j)
        if df is None or df.empty:
            _INSTI_LAST_ERROR["value"] = (
                f"TPEX 無資料 date={date_roc} keys={list(j.keys())[:5]}"
            )
            return None
        return df
    except requests.exceptions.Timeout:
        _INSTI_LAST_ERROR["value"] = "TPEX 連線逾時 (8秒)"
        return None
    except Exception as e:
        _INSTI_LAST_ERROR["value"] = f"TPEX 錯誤: {type(e).__name__}: {e}"
        return None


def _match_stock_row(df, stock_code):
    """從 DataFrame 找出代號對應列 (支援多種欄位名)。"""
    if df is None or df.empty:
        return None
    code_col = None
    for c in df.columns:
        cs = str(c)
        if '證券代號' in cs or '股票代號' in cs or '代號' in cs:
            code_col = c
            break
    if code_col is None:
        return None
    mask = df[code_col].astype(str).str.strip() == str(stock_code).strip()
    match = df[mask]
    return match.iloc[0] if not match.empty else None


def _find_col(row, keywords):
    """找第一個欄位名包含任一 keyword 的值 (int)。"""
    for col in row.index:
        cs = str(col)
        for kw in keywords:
            if kw in cs:
                return _parse_int(row[col])
    return 0


def get_institutional_data(stock_code, is_otc=False, days=3):
    """取得真實三大法人資料 (TWSE / TPEX),預設抓近 3 個交易日。
    回傳: DataFrame 含 [日期, 外資買賣, 投信買賣, 自營商買賣, 融資增減, 融券增減] (單位: 張)
          取不到則回傳 None。
    """
    _INSTI_LAST_ERROR["value"] = None
    code = str(stock_code).strip()
    if not code.isdigit():
        _INSTI_LAST_ERROR["value"] = "非台股代號 (僅台股有法人資料)"
        return None

    rows = []
    for dt in _recent_trading_dates(days):
        foreign = trust = dealer = 0
        found = False

        if is_otc:
            df_i = _fetch_tpex_insti_day(_roc_date(dt))
            row = _match_stock_row(df_i, code) if df_i is not None else None
            if row is not None:
                use_shares = any('股數' in str(c) for c in row.index)
                foreign = _find_col(row, ['外資及陸資', '外陸資', '外資'])
                trust = _find_col(row, ['投信'])
                dealer = _find_col(row, ['自營商'])
                if use_shares:
                    foreign //= 1000
                    trust //= 1000
                    dealer //= 1000
                found = True
        else:
            df_i = _fetch_twse_insti_day(dt.strftime('%Y%m%d'))
            row = _match_stock_row(df_i, code) if df_i is not None else None
            if row is not None:
                f1 = _find_col(row, ['外陸資買賣超股數(不含外資自營商)', '外陸資買賣超'])
                f2 = _find_col(row, ['外資自營商買賣超股數'])
                foreign = (f1 + f2) // 1000  # 股 → 張
                trust = _find_col(row, ['投信買賣超股數', '投信買賣超']) // 1000
                dealer = _find_col(row, ['自營商買賣超股數', '自營商買賣超']) // 1000
                found = True

        if found:
            rows.append({
                '日期': dt,
                '外資買賣': foreign,
                '投信買賣': trust,
                '自營商買賣': dealer,
                '融資增減': 0,
                '融券增減': 0,
            })

    if not rows:
        if _INSTI_LAST_ERROR["value"] is None:
            _INSTI_LAST_ERROR["value"] = f"找不到 {code} 的法人資料 (可能非當前可查詢範圍)"
        return None
    out = pd.DataFrame(rows).sort_values('日期').reset_index(drop=True)
    return out


def get_insti_last_error():
    """取得最近一次法人資料抓取失敗原因 (供 UI 顯示)。"""
    return _INSTI_LAST_ERROR.get("value")

# ==================== 財報分析函數 ====================

def get_fundamental_data(ticker_symbol):
    """獲取並計算財務指標 (含毛利率/淨利率/ROE/P/B/股息率/營收年增等)"""
    try:
        ticker = yf.Ticker(ticker_symbol)

        # 年報優先,年報無資料時改用季報 (台股常常年報要很久才更新)
        income_stmt = ticker.financials
        if income_stmt is None or income_stmt.empty:
            try:
                income_stmt = ticker.quarterly_financials
            except Exception:
                income_stmt = None

        balance_sheet = ticker.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            try:
                balance_sheet = ticker.quarterly_balance_sheet
            except Exception:
                balance_sheet = None

        info = ticker.info or {}

        if income_stmt is None or income_stmt.empty or balance_sheet is None or balance_sheet.empty:
            return None

        latest_income = income_stmt.iloc[:, 0]
        latest_balance = balance_sheet.iloc[:, 0]
        prev_income = income_stmt.iloc[:, 1] if income_stmt.shape[1] > 1 else None

        metrics = {}

        # 報表日期
        try:
            metrics['報表期別'] = str(income_stmt.columns[0])[:10]
        except:
            metrics['報表期別'] = 'N/A'

        # 1. 毛利率
        try:
            gross_profit = latest_income.get('Gross Profit', 0)
            total_revenue = latest_income.get('Total Revenue', 1)
            metrics['毛利率'] = (gross_profit / total_revenue * 100) if total_revenue else 0
        except:
            metrics['毛利率'] = 0

        # 2. 淨利率
        try:
            net_income = latest_income.get('Net Income', 0)
            total_revenue = latest_income.get('Total Revenue', 1)
            metrics['淨利率'] = (net_income / total_revenue * 100) if total_revenue else 0
        except:
            metrics['淨利率'] = 0

        # 3. 總資產週轉率
        try:
            total_revenue = latest_income.get('Total Revenue', 0)
            total_assets = latest_balance.get('Total Assets', 1)
            metrics['總資產週轉率'] = (total_revenue / total_assets) if total_assets else 0
        except:
            metrics['總資產週轉率'] = 0

        # 4. 權益乘數
        try:
            total_assets = latest_balance.get('Total Assets', 0)
            total_equity = latest_balance.get('Stockholders Equity', 1)
            metrics['權益乘數'] = (total_assets / total_equity) if total_equity else 0
        except:
            metrics['權益乘數'] = 0

        # 5. 每股淨值 (BVPS)
        try:
            total_equity = latest_balance.get('Stockholders Equity', 0)
            shares_outstanding = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding') or 0
            metrics['每股淨值'] = (total_equity / shares_outstanding) if shares_outstanding > 0 else 0
        except:
            metrics['每股淨值'] = 0

        # 6. 每股盈餘 (EPS) - 多層 fallback
        try:
            eps = info.get('trailingEps', None)
            if eps is None or eps == 0:
                eps = info.get('forwardEps', None)
            if eps is None or eps == 0:
                net_income = latest_income.get('Net Income', 0)
                shares_outstanding = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding') or 0
                eps = (net_income / shares_outstanding) if shares_outstanding else 0
            metrics['每股盈餘'] = eps if eps is not None else 0
        except:
            metrics['每股盈餘'] = 0

        # 7. ROE
        try:
            net_income = latest_income.get('Net Income', 0)
            total_equity = latest_balance.get('Stockholders Equity', 1)
            metrics['ROE'] = (net_income / total_equity * 100) if total_equity else 0
        except:
            metrics['ROE'] = 0

        # 8. 股價淨值比 (P/B)
        try:
            pb = info.get('priceToBook', None)
            if pb is None and metrics['每股淨值']:
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if current_price:
                    pb = current_price / metrics['每股淨值']
            metrics['股價淨值比'] = pb if pb is not None else 0
        except:
            metrics['股價淨值比'] = 0

        # 9. 股息殖利率
        try:
            dy = info.get('dividendYield', None)
            # yfinance 有時回傳 0.03 (3%)，有時回傳 3.0；統一處理為 %
            if dy is not None:
                metrics['股息殖利率'] = dy * 100 if dy < 1 else dy
            else:
                metrics['股息殖利率'] = 0
        except:
            metrics['股息殖利率'] = 0

        # 10. 營收年增率 (YoY)
        try:
            if prev_income is not None:
                curr_rev = latest_income.get('Total Revenue', 0)
                prev_rev = prev_income.get('Total Revenue', 0)
                if prev_rev:
                    metrics['營收年增率'] = (curr_rev - prev_rev) / abs(prev_rev) * 100
                else:
                    metrics['營收年增率'] = 0
            else:
                metrics['營收年增率'] = 0
        except:
            metrics['營收年增率'] = 0

        return metrics

    except Exception:
        return None

def calculate_pe_range(ticker_symbol, eps):
    """計算近兩年本益比區間"""
    try:
        if eps <= 0:
            return None
        
        ticker = yf.Ticker(ticker_symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None
        
        hist['PE_Ratio'] = hist['Close'] / eps
        
        pe_stats = {
            '最高本益比': hist['PE_Ratio'].max(),
            '平均本益比': hist['PE_Ratio'].mean(),
            '最低本益比': hist['PE_Ratio'].min(),
            '當前股價': hist['Close'].iloc[-1]
        }
        
        return pe_stats
        
    except:
        return None

def create_valuation_table(eps, pe_stats):
    """建立估值推算表格"""
    try:
        valuation_data = {
            '估值情境': ['樂觀價', '合理價', '悲觀價'],
            '採用本益比': [
                pe_stats['最高本益比'],
                pe_stats['平均本益比'],
                pe_stats['最低本益比']
            ],
            '基本面EPS': [eps, eps, eps],
            '推算股價': [
                eps * pe_stats['最高本益比'],
                eps * pe_stats['平均本益比'],
                eps * pe_stats['最低本益比']
            ]
        }
        
        df = pd.DataFrame(valuation_data)
        df['採用本益比'] = df['採用本益比'].round(2)
        df['基本面EPS'] = df['基本面EPS'].round(2)
        df['推算股價'] = df['推算股價'].round(2)
        
        return df
        
    except:
        return None

def evaluate_stock_valuation(current_price, optimistic, fair, pessimistic):
    """判斷股票估值位階"""
    if current_price <= pessimistic:
        return "非常便宜", "cheap", "🟢 目前股價低於悲觀價，處於超值區間"
    elif current_price <= fair:
        return "便宜", "cheap", "🟢 目前股價低於合理價，值得關注"
    elif current_price <= fair * 1.1:
        return "合理", "fair", "🟡 目前股價接近合理價，可以持有"
    elif current_price <= optimistic:
        return "偏貴", "fair", "🟡 目前股價接近樂觀價，建議謹慎"
    else:
        return "昂貴", "expensive", "🔴 目前股價高於樂觀價，估值偏高"

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
st.markdown("**技術面 × 籌碼面 × 基本面 × 智能篩選**")

# 側邊欄: 資料源連線診斷
with st.sidebar:
    st.markdown("### 🔧 系統診斷")
    st.caption("若查詢時無法取得資料,可點下方按鈕確認各資料源是否能連線。")
    if st.button("🩺 測試資料源連線"):
        diag_results = []
        # yfinance
        try:
            import time as _time
            t0 = _time.time()
            _probe = yf.Ticker("2330.TW").history(period="5d")
            ok = not _probe.empty
            diag_results.append(("yfinance (股價/財報)", ok, f"{_time.time()-t0:.1f}s"))
        except Exception as e:
            diag_results.append(("yfinance (股價/財報)", False, type(e).__name__))
        # TWSE
        try:
            import time as _time
            t0 = _time.time()
            r = requests.get("https://www.twse.com.tw/rwd/zh/fund/T86",
                             params={"date": "20250415", "selectType": "ALL", "response": "json"},
                             headers=_HTTP_HEADERS, timeout=8)
            diag_results.append(("TWSE (上市法人)", r.status_code == 200, f"HTTP {r.status_code} ({_time.time()-t0:.1f}s)"))
        except Exception as e:
            diag_results.append(("TWSE (上市法人)", False, type(e).__name__))
        # TPEX
        try:
            import time as _time
            t0 = _time.time()
            r = requests.get("https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
                             params={"type": "Daily", "sect": "AL", "date": "114/04/15",
                                     "id": "", "response": "json"},
                             headers=_HTTP_HEADERS, timeout=8)
            diag_results.append(("TPEX (上櫃法人)", r.status_code == 200, f"HTTP {r.status_code} ({_time.time()-t0:.1f}s)"))
        except Exception as e:
            diag_results.append(("TPEX (上櫃法人)", False, type(e).__name__))

        for name, ok, detail in diag_results:
            if ok:
                st.success(f"✅ {name}: {detail}")
            else:
                st.error(f"❌ {name}: {detail}")

    st.markdown("---")
    st.caption("下方按鈕會抓最近 3 個交易日的 2330 法人資料,並顯示原始欄位供除錯。")
    if st.button("🔍 實測抓 2330 法人資料"):
        # 直接呼叫 production 函式,看是否成功
        _INSTI_LAST_ERROR["value"] = None
        test_df = get_institutional_data("2330", is_otc=False, days=3)
        if test_df is not None and not test_df.empty:
            st.success("✅ 法人資料抓取成功")
            st.dataframe(test_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"❌ 抓取失敗: {get_insti_last_error() or '未知'}")

        # 額外顯示 TWSE 原始 JSON 的 top-level keys + 第一個 table 的 fields
        st.markdown("##### 🔬 TWSE 原始回傳結構 (供除錯)")
        try:
            probe_dates = _recent_trading_dates(1)
            if probe_dates:
                ymd = probe_dates[0].strftime('%Y%m%d')
                r = requests.get(
                    "https://www.twse.com.tw/rwd/zh/fund/T86",
                    params={"date": ymd, "selectType": "ALL", "response": "json"},
                    headers={**_HTTP_HEADERS, "Referer": "https://www.twse.com.tw/"},
                    timeout=8,
                )
                if r.status_code == 200:
                    j = r.json()
                    st.code(f"date tried: {ymd}\ntop-level keys: {list(j.keys())}\n"
                            f"stat: {j.get('stat')}\n"
                            f"top fields count: {len(j.get('fields') or [])}\n"
                            f"top data rows: {len(j.get('data') or [])}\n"
                            f"tables count: {len(j.get('tables') or [])}\n"
                            + (f"tables[0] fields (前6): {[str(x)[:30] for x in (j.get('tables')[0].get('fields') or [])[:6]]}\n"
                               f"tables[0] data rows: {len(j.get('tables')[0].get('data') or [])}"
                               if j.get('tables') else ""))
                else:
                    st.code(f"HTTP {r.status_code}")
        except Exception as e:
            st.code(f"錯誤: {type(e).__name__}: {e}")

tab1, tab2, tab3 = st.tabs(["🔍 單股分析", "🎯 智能選股", "📊 批次掃描"])

# ==================== Tab 1: 單股完整分析 ====================
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        stock_code = st.text_input(
            "🔢 輸入股票代號",
            value="2330",
            help="台股上市:2330 | 上櫃/興櫃:6488 (自動加 .TWO) | 美股:AAPL, TSLA, NVDA",
        )

    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 完整分析", type="primary", use_container_width=True)

    st.caption("💡 支援市場: 台股上市(.TW)、上櫃/興櫃(.TWO)、美股 (直接輸入英文代號如 AAPL)")
    
    if analyze_btn:
        stock_code = str(stock_code).strip().upper()
        with st.spinner("分析中..."):
            df, stock, ticker_symbol = get_stock_data(stock_code)

            if df is not None and not df.empty:
                df = calc_indicators(df)
                is_otc = bool(ticker_symbol and ticker_symbol.endswith('.TWO'))
                with st.spinner("載入法人買賣資料 (TWSE/TPEX)..."):
                    inst_df = get_institutional_data(stock_code, is_otc=is_otc)

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

                    if inst_df is not None and not inst_df.empty:
                        st.caption(f"資料來源: {'TPEX 櫃買中心' if is_otc else 'TWSE 證交所'} (真實)")

                        if inst_signals:
                            for key, (desc, signal_type, score) in inst_signals.items():
                                if signal_type == 'BUY':
                                    st.success(f"**{key}:** {desc} (+{score}分)")
                                elif signal_type == 'SELL':
                                    st.error(f"**{key}:** {desc} ({score}分)")
                                else:
                                    st.info(f"**{key}:** {desc}")

                        st.markdown("##### 近日法人買賣 (單位: 張)")
                        display_df = inst_df.copy()
                        display_df['日期'] = pd.to_datetime(display_df['日期']).dt.strftime('%Y-%m-%d')
                        st.dataframe(
                            display_df[['日期', '外資買賣', '投信買賣', '自營商買賣']].tail(5),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        if str(stock_code).strip().isdigit():
                            err = get_insti_last_error() or "未知原因"
                            st.warning(f"⚠️ 無法取得真實籌碼資料\n\n原因: {err}")
                            st.caption("💡 若為假日或剛開盤,TWSE 資料通常於 15:00 後公布。若持續失敗,可能是雲端伺服器無法存取 TWSE,請告知以改用其他資料源。")
                        else:
                            st.info("ℹ️ 美股/非台股不提供籌碼面分析")
                
                st.markdown("---")
                
                # ==================== 基本面分析 ====================
                st.subheader("📊 基本面分析與估值")

                with st.spinner("載入財報資料..."):
                    fundamental_data = get_fundamental_data(ticker_symbol)

                if fundamental_data and fundamental_data['每股盈餘'] != 0:
                    report_period = fundamental_data.get('報表期別', 'N/A')
                    st.caption(f"📅 財報期別: {report_period} | 代號: {ticker_symbol}")

                    # 財務指標
                    st.markdown("##### 💼 財務指標 (最新年度)")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("毛利率", f"{fundamental_data['毛利率']:.2f}%")
                        st.metric("淨利率", f"{fundamental_data['淨利率']:.2f}%")

                    with col2:
                        st.metric("總資產週轉率", f"{fundamental_data['總資產週轉率']:.2f}")
                        st.metric("權益乘數", f"{fundamental_data['權益乘數']:.2f}")

                    with col3:
                        st.metric("每股淨值", f"${fundamental_data['每股淨值']:.2f}")
                        st.metric("EPS", f"${fundamental_data['每股盈餘']:.2f}")

                    # 延伸指標
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pb = fundamental_data.get('股價淨值比', 0)
                        st.metric("股價淨值比 (P/B)", f"{pb:.2f}" if pb else "N/A")
                    with col2:
                        dy = fundamental_data.get('股息殖利率', 0)
                        st.metric("股息殖利率", f"{dy:.2f}%" if dy else "N/A")
                    with col3:
                        yoy = fundamental_data.get('營收年增率', 0)
                        st.metric("營收年增率", f"{yoy:+.2f}%" if yoy else "N/A")

                    st.info(f"📈 **ROE:** {fundamental_data['ROE']:.2f}%")
                    
                    st.markdown("---")
                    
                    # 本益比估值
                    st.markdown("##### 💰 本益比估值 (近兩年)")
                    
                    eps = fundamental_data['每股盈餘']
                    
                    if eps > 0:
                        with st.spinner("計算本益比區間..."):
                            pe_stats = calculate_pe_range(ticker_symbol, eps)
                        
                        if pe_stats:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("最高本益比", f"{pe_stats['最高本益比']:.2f}")
                            with col2:
                                st.metric("平均本益比", f"{pe_stats['平均本益比']:.2f}")
                            with col3:
                                st.metric("最低本益比", f"{pe_stats['最低本益比']:.2f}")
                            with col4:
                                current_pe = latest['Close'] / eps
                                st.metric("當前本益比", f"{current_pe:.2f}")
                            
                            st.markdown("---")
                            st.markdown("##### 📋 估值推算")
                            
                            valuation_df = create_valuation_table(eps, pe_stats)
                            
                            if valuation_df is not None:
                                st.dataframe(valuation_df, use_container_width=True, hide_index=True)
                                
                                optimistic = valuation_df.loc[0, '推算股價']
                                fair = valuation_df.loc[1, '推算股價']
                                pessimistic = valuation_df.loc[2, '推算股價']
                                
                                evaluation, eval_type, message = evaluate_stock_valuation(
                                    latest['Close'], optimistic, fair, pessimistic
                                )
                                
                                st.markdown("---")
                                
                                if eval_type == "cheap":
                                    st.markdown(f"""
                                    <div class='valuation-cheap'>
                                        <h4 style='color: #2E7D32; margin:0;'>🎯 {evaluation}</h4>
                                        <p style='margin:8px 0 0 0;'>{message}</p>
                                        <p style='margin:5px 0 0 0; font-size:14px;'>當前: ${latest['Close']:.2f} | 悲觀: ${pessimistic:.2f} | 合理: ${fair:.2f} | 樂觀: ${optimistic:.2f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif eval_type == "fair":
                                    st.markdown(f"""
                                    <div class='valuation-fair'>
                                        <h4 style='color: #F57C00; margin:0;'>🎯 {evaluation}</h4>
                                        <p style='margin:8px 0 0 0;'>{message}</p>
                                        <p style='margin:5px 0 0 0; font-size:14px;'>當前: ${latest['Close']:.2f} | 悲觀: ${pessimistic:.2f} | 合理: ${fair:.2f} | 樂觀: ${optimistic:.2f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class='valuation-expensive'>
                                        <h4 style='color: #C62828; margin:0;'>🎯 {evaluation}</h4>
                                        <p style='margin:8px 0 0 0;'>{message}</p>
                                        <p style='margin:5px 0 0 0; font-size:14px;'>當前: ${latest['Close']:.2f} | 悲觀: ${pessimistic:.2f} | 合理: ${fair:.2f} | 樂觀: ${optimistic:.2f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ 無法計算本益比區間 (歷史數據不足)")
                    else:
                        st.warning("⚠️ EPS ≤ 0，無法進行本益比估值")
                elif fundamental_data:
                    st.info(f"💡 已取得部分財報資料，但 EPS 數據不完整 (代號: {ticker_symbol})")
                    st.caption("建議: 可能是財報尚未公布或數據延遲,請稍後再試")
                else:
                    st.info(f"💡 暫無完整財報數據 (代號: {ticker_symbol})")
                    st.caption("可能原因: ETF、財報未公開、或 yfinance 無該股票財報資料")
                    st.caption("💡 提示: 美股請直接輸入代號 (例: AAPL)，台股會自動加上 .TW")
            
            else:
                st.error(f"❌ 無法取得 {stock_code} 的資料")

# ==================== Tab 2: 智能選股 ====================
with tab2:
    st.subheader("🎯 自訂條件選股")
    st.markdown("勾選技術面 + 基本面 + 籌碼面條件,系統自動找出符合的股票 (支援台股、上櫃/興櫃、美股)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📈 技術面條件")

        tech_conditions = []

        if st.checkbox("✅ 多頭排列 (價 > MA5 > MA10 > MA20)"):
            tech_conditions.append("多頭排列")

        if st.checkbox("✅ MA黃金交叉"):
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
        st.markdown("#### 📊 基本面條件")
        st.caption("資料來源: yfinance 財報 (真實數據)")

        fund_conditions = []

        if st.checkbox("✅ EPS > 0 (有獲利)"):
            fund_conditions.append(("EPS_positive", 0))

        roe_thresh = st.number_input("ROE ≥ (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0, key="roe_t")
        if st.checkbox(f"✅ ROE ≥ {roe_thresh:.0f}%"):
            fund_conditions.append(("ROE", roe_thresh))

        gm_thresh = st.number_input("毛利率 ≥ (%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0, key="gm_t")
        if st.checkbox(f"✅ 毛利率 ≥ {gm_thresh:.0f}%"):
            fund_conditions.append(("毛利率", gm_thresh))

        nm_thresh = st.number_input("淨利率 ≥ (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="nm_t")
        if st.checkbox(f"✅ 淨利率 ≥ {nm_thresh:.0f}%"):
            fund_conditions.append(("淨利率", nm_thresh))

        pb_thresh = st.number_input("P/B ≤", min_value=0.1, max_value=50.0, value=3.0, step=0.5, key="pb_t")
        if st.checkbox(f"✅ 股價淨值比 ≤ {pb_thresh:.1f}"):
            fund_conditions.append(("股價淨值比_max", pb_thresh))

        dy_thresh = st.number_input("股息殖利率 ≥ (%)", min_value=0.0, max_value=30.0, value=3.0, step=0.5, key="dy_t")
        if st.checkbox(f"✅ 股息殖利率 ≥ {dy_thresh:.1f}%"):
            fund_conditions.append(("股息殖利率", dy_thresh))

        if st.checkbox("✅ 營收年增率 > 0"):
            fund_conditions.append(("營收年增率", 0))

        if st.checkbox("✅ 估值判斷: 便宜 或 合理"):
            fund_conditions.append(("估值", "cheap_fair"))

    with col3:
        st.markdown("#### 💼 籌碼面條件")
        st.caption("資料來源: TWSE (上市) / TPEX (上櫃) 真實法人資料")

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
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        stock_pool = st.selectbox(
            "選擇股票池",
            ["台灣50成分股", "電子股", "金融股", "傳產股", "美股科技股", "美股道瓊30", "自訂清單"]
        )

    with col2:
        logic_mode = st.radio("條件邏輯", ["任一符合 (OR)", "全部符合 (AND)"], index=0)

    with col3:
        min_score = st.slider("最低總分", -10, 20, -5)

    with col4:
        st.write("")
        st.write("")
        search_btn = st.button("🔍 開始搜尋", type="primary", use_container_width=True)

    # 自訂清單
    if stock_pool == "自訂清單":
        custom_stocks = st.text_area(
            "輸入股票代號 (每行一個;台股數字 / 美股英文如 AAPL)",
            value="2330\n2317\n2454\nAAPL\nNVDA",
            height=100
        )
    
    if search_btn:
        # 準備股票清單 - 擴大股票池
        if stock_pool == "台灣50成分股":
            stock_list = ["2330", "2317", "2454", "2308", "2382", "2881", "2886", "2412", "2303", "1301",
                          "1303", "2891", "2002", "2884", "2892", "2912", "2882", "2395", "2207", "2887",
                          "2357", "3711", "2379", "5880", "2301", "2408", "3008", "2892", "2385", "2409"]
        elif stock_pool == "電子股":
            stock_list = ["2330", "2317", "2454", "2308", "2382", "3711", "2357", "3034", "2327", "2345",
                          "2303", "2379", "3037", "2474", "2409", "3231", "6505", "2344", "2324", "2352",
                          "3481", "2377", "2458", "3443", "2360", "4938", "2393", "2376", "6239", "3665"]
        elif stock_pool == "金融股":
            stock_list = ["2881", "2882", "2883", "2884", "2885", "2886", "2887", "2890", "2891", "2892",
                          "2880", "2888", "2889", "5880", "2834", "2836", "2845", "2849", "2867", "2897"]
        elif stock_pool == "傳產股":
            stock_list = ["1301", "1303", "2002", "2207", "2408", "2409", "2912", "5880", "6505", "9904",
                          "1101", "1102", "1216", "2105", "2201", "2301", "2395", "2412", "9910", "1326"]
        elif stock_pool == "美股科技股":
            stock_list = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",
                          "AMD", "ADBE", "NFLX", "INTC", "QCOM", "CSCO", "IBM", "TXN", "NOW", "PLTR"]
        elif stock_pool == "美股道瓊30":
            stock_list = ["AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "UNH", "HD", "MA",
                          "CVX", "KO", "MRK", "AXP", "MCD", "CSCO", "CAT", "IBM", "CRM", "GS",
                          "VZ", "HON", "AMGN", "DIS", "NKE", "BA", "MMM", "TRV", "DOW", "WBA"]
        else:
            stock_list = [s.strip() for s in custom_stocks.split('\n') if s.strip()]

        # 檢查是否有選擇條件
        has_conditions = len(tech_conditions) + len(fund_conditions) + len(inst_conditions) > 0
        use_and = logic_mode.startswith("全部")

        need_fundamental = len(fund_conditions) > 0
        fund_note = " + 載入財報" if need_fundamental else ""
        logic_text = "全部符合" if use_and else "任一符合"
        st.info(
            f"搜尋中... 股票池: {len(stock_list)}檔 | "
            f"條件: {logic_text if has_conditions else '無條件限制'}{fund_note}"
        )
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, code in enumerate(stock_list):
            status.text(f"分析: {code} ({i+1}/{len(stock_list)})")
            progress.progress((i + 1) / len(stock_list))
            
            try:
                df, stock, ticker_used = get_stock_data(code, period="3mo")
                if df is None or len(df) < 60:
                    continue

                df = calc_indicators(df)
                is_otc_code = bool(ticker_used and ticker_used.endswith('.TWO'))
                inst_df = get_institutional_data(code, is_otc=is_otc_code) if len(inst_conditions) > 0 else None

                tech_signals, tech_score = analyze_technical(df)
                inst_signals, inst_score = analyze_institutional(inst_df)
                recommendation, action, total_score = get_final_recommendation(tech_score, inst_score)

                latest = df.iloc[-1]

                # 逐一檢查所有條件,記錄是否滿足
                condition_results = []  # [(cond_label, passed)]

                # --- 技術面條件 ---
                if "多頭排列" in tech_conditions:
                    passed = '均線' in tech_signals and tech_signals['均線'][0] == '多頭排列'
                    condition_results.append(("多頭排列", passed))
                if "MA黃金交叉" in tech_conditions:
                    condition_results.append(("MA黃金交叉", 'MA交叉' in tech_signals))
                if "MACD紅柱" in tech_conditions:
                    passed = 'MACD' in tech_signals and tech_signals['MACD'][1] == 'BUY'
                    condition_results.append(("MACD紅柱", passed))
                if "RSI超賣" in tech_conditions:
                    passed = 'RSI' in tech_signals and tech_signals['RSI'][0] == '超賣'
                    condition_results.append(("RSI超賣", passed))
                if "KD低檔" in tech_conditions:
                    passed = 'KD' in tech_signals and tech_signals['KD'][0] == '超賣'
                    condition_results.append(("KD低檔", passed))
                if "放量" in tech_conditions:
                    passed = '成交量' in tech_signals and '上漲' in tech_signals['成交量'][0]
                    condition_results.append(("放量上漲", passed))
                if "站上MA20" in tech_conditions:
                    condition_results.append(("站上MA20", latest['Close'] > latest['MA20']))

                # --- 籌碼面條件 (TWSE/TPEX 真實資料) ---
                if "外資買超" in inst_conditions:
                    passed = '外資' in inst_signals and inst_signals['外資'][1] == 'BUY'
                    condition_results.append(("外資買超", passed))
                if "投信買超" in inst_conditions:
                    passed = '投信' in inst_signals and inst_signals['投信'][1] == 'BUY'
                    condition_results.append(("投信買超", passed))
                if "自營商買超" in inst_conditions:
                    passed = '自營商' in inst_signals and inst_signals['自營商'][1] == 'BUY'
                    condition_results.append(("自營商買超", passed))
                if "三法人買超" in inst_conditions:
                    passed = ('外資' in inst_signals and inst_signals['外資'][1] == 'BUY' and
                              '投信' in inst_signals and inst_signals['投信'][1] == 'BUY' and
                              '自營商' in inst_signals and inst_signals['自營商'][1] == 'BUY')
                    condition_results.append(("三法人買超", passed))
                if "融資券雙降" in inst_conditions:
                    passed = '融資券' in inst_signals and '雙降' in inst_signals['融資券'][0]
                    condition_results.append(("融資券雙降", passed))

                # --- 基本面條件 (真實財報資料) ---
                fund_data = None
                fund_eval_type = None
                if need_fundamental:
                    fund_data = get_fundamental_data(ticker_used)
                    # 估值判斷需要 PE 區間
                    if fund_data and any(c[0] == "估值" for c in fund_conditions):
                        eps = fund_data.get('每股盈餘', 0)
                        if eps and eps > 0:
                            pe_stats = calculate_pe_range(ticker_used, eps)
                            if pe_stats:
                                optimistic = eps * pe_stats['最高本益比']
                                fair = eps * pe_stats['平均本益比']
                                pessimistic = eps * pe_stats['最低本益比']
                                _, fund_eval_type, _ = evaluate_stock_valuation(
                                    latest['Close'], optimistic, fair, pessimistic
                                )

                for cond, threshold in fund_conditions:
                    if fund_data is None:
                        condition_results.append((f"{cond}(無財報)", False))
                        continue
                    if cond == "EPS_positive":
                        condition_results.append(("EPS>0", fund_data.get('每股盈餘', 0) > 0))
                    elif cond == "ROE":
                        condition_results.append((f"ROE≥{threshold:.0f}%", fund_data.get('ROE', 0) >= threshold))
                    elif cond == "毛利率":
                        condition_results.append((f"毛利率≥{threshold:.0f}%", fund_data.get('毛利率', 0) >= threshold))
                    elif cond == "淨利率":
                        condition_results.append((f"淨利率≥{threshold:.0f}%", fund_data.get('淨利率', 0) >= threshold))
                    elif cond == "股價淨值比_max":
                        pb_val = fund_data.get('股價淨值比', 0)
                        condition_results.append(
                            (f"P/B≤{threshold:.1f}", pb_val > 0 and pb_val <= threshold)
                        )
                    elif cond == "股息殖利率":
                        condition_results.append((f"股息≥{threshold:.1f}%", fund_data.get('股息殖利率', 0) >= threshold))
                    elif cond == "營收年增率":
                        condition_results.append(("營收年增>0", fund_data.get('營收年增率', 0) > 0))
                    elif cond == "估值":
                        condition_results.append(("估值=便宜/合理", fund_eval_type in ("cheap", "fair")))

                # 綜合判斷 (AND 或 OR)
                if not has_conditions:
                    match = True
                    matched_conditions = ["(無條件,僅分數過濾)"]
                elif use_and:
                    match = len(condition_results) > 0 and all(p for _, p in condition_results)
                    matched_conditions = [lbl for lbl, p in condition_results if p]
                else:
                    match = any(p for _, p in condition_results)
                    matched_conditions = [lbl for lbl, p in condition_results if p]

                # 套用最低總分過濾
                if match and total_score < min_score:
                    match = False

                if match:
                    try:
                        name = stock.info.get('longName', code)
                    except:
                        name = code

                    matched_text = ", ".join(matched_conditions) if matched_conditions else "達到分數標準"

                    row = {
                        '代號': code,
                        '名稱': name,
                        '建議': recommendation,
                        '符合條件': matched_text,
                        '總分': total_score,
                        '技術分': tech_score,
                        '籌碼分': inst_score,
                        '收盤價': f"${latest['Close']:.2f}",
                        'RSI': f"{latest['RSI']:.1f}",
                    }
                    if fund_data:
                        row['EPS'] = f"{fund_data.get('每股盈餘', 0):.2f}"
                        row['ROE%'] = f"{fund_data.get('ROE', 0):.1f}"
                        row['毛利率%'] = f"{fund_data.get('毛利率', 0):.1f}"
                        row['P/B'] = f"{fund_data.get('股價淨值比', 0):.2f}"
                    results.append(row)

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
    st.caption("支援混合輸入: 台股 (2330)、上櫃/興櫃 (6488)、美股 (AAPL)")

    stock_input = st.text_area(
        "輸入股票代號 (每行一個)",
        value="2330\n2317\n2454\n3008\n2308\nAAPL\nNVDA\nTSLA",
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
                df, stock, ticker_used = get_stock_data(code, period="3mo")
                if df is None or len(df) < 60:
                    continue

                df = calc_indicators(df)
                is_otc_code = bool(ticker_used and ticker_used.endswith('.TWO'))
                inst_df = get_institutional_data(code, is_otc=is_otc_code)

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
st.caption("⚠️ 本系統僅供參考,投資有風險。資料來源: yfinance (股價/財報) + TWSE/TPEX (法人籌碼)。")
