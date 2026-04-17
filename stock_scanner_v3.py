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
    """清理歷史資料:去掉 Close 為 NaN 的列 (盤中 yfinance 會回傳半完整列)。"""
    if df is None or df.empty:
        return df
    return df.dropna(subset=['Close'])


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_twse_stock_month(stock_code, yyyymm):
    """抓 TWSE 個股某月日成交資料 (權威來源,含今日)。回傳 DataFrame 或 None。"""
    try:
        url = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
        params = {"stockNo": stock_code, "date": f"{yyyymm}01", "response": "json"}
        headers = dict(_HTTP_HEADERS)
        headers["Referer"] = "https://www.twse.com.tw/"
        r = requests.get(url, params=params, timeout=8, headers=headers)
        if r.status_code != 200:
            return None
        j = r.json()
        rows_raw = j.get('data') or []
        if not rows_raw:
            return None

        def _f(x):
            s = str(x).replace(',', '').strip()
            try:
                return float(s) if s not in ('', '-', '--', 'X') else None
            except Exception:
                return None

        rows = []
        for row in rows_raw:
            try:
                parts = str(row[0]).split('/')
                if len(parts) != 3:
                    continue
                dt = datetime(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))
                volume = _parse_int(row[1])  # 股數
                open_p = _f(row[3])
                high = _f(row[4])
                low = _f(row[5])
                close = _f(row[6])
                if close is None:
                    continue
                rows.append({
                    'Date': dt,
                    'Open': open_p if open_p is not None else close,
                    'High': high if high is not None else close,
                    'Low': low if low is not None else close,
                    'Close': close,
                    'Volume': volume,
                })
            except Exception:
                continue
        if not rows:
            return None
        return pd.DataFrame(rows).set_index('Date').sort_index()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_tpex_stock_month(stock_code, yyyymm):
    """抓 TPEX 上櫃個股某月日成交資料。回傳 DataFrame 或 None。"""
    try:
        # TPEX 月份參數是 ROC 年/月: 114/04
        year = int(yyyymm[:4]) - 1911
        month = int(yyyymm[4:6])
        roc_date = f"{year}/{month:02d}"
        url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
        params = {"code": stock_code, "date": roc_date, "id": "", "response": "json"}
        headers = dict(_HTTP_HEADERS)
        headers["Referer"] = "https://www.tpex.org.tw/"
        r = requests.get(url, params=params, timeout=8, headers=headers)
        if r.status_code != 200:
            return None
        j = r.json()
        df_raw = _df_from_any(j)
        if df_raw is None or df_raw.empty:
            return None

        def _f(x):
            s = str(x).replace(',', '').strip()
            try:
                return float(s) if s not in ('', '-', '--', 'X') else None
            except Exception:
                return None

        rows = []
        cols = list(df_raw.columns)
        for _, row in df_raw.iterrows():
            try:
                date_str = str(row[cols[0]])
                parts = date_str.split('/')
                if len(parts) != 3:
                    continue
                dt = datetime(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))
                # TPEX 欄位順序: 日期, 成交仟股, 成交仟元, 開盤, 最高, 最低, 收盤, 漲跌, 筆數
                volume = _parse_int(row[cols[1]]) * 1000 if len(cols) > 1 else 0
                open_p = _f(row[cols[3]]) if len(cols) > 3 else None
                high = _f(row[cols[4]]) if len(cols) > 4 else None
                low = _f(row[cols[5]]) if len(cols) > 5 else None
                close = _f(row[cols[6]]) if len(cols) > 6 else None
                if close is None:
                    continue
                rows.append({
                    'Date': dt, 'Open': open_p if open_p is not None else close,
                    'High': high if high is not None else close,
                    'Low': low if low is not None else close,
                    'Close': close, 'Volume': volume,
                })
            except Exception:
                continue
        if not rows:
            return None
        return pd.DataFrame(rows).set_index('Date').sort_index()
    except Exception:
        return None


def _merge_with_twse(df, ticker):
    """若 yfinance 資料落後超過 1 天,用 TWSE/TPEX 補上最新日。"""
    if not ticker.endswith(('.TW', '.TWO')):
        return df
    stock_code = ticker.replace('.TW', '').replace('.TWO', '')

    # 計算 yfinance 最新日期與今日差距
    try:
        today = datetime.now().date()
        if df is not None and not df.empty:
            last = df.index[-1]
            last_date = last.date() if hasattr(last, 'date') else pd.Timestamp(last).date()
        else:
            last_date = today - timedelta(days=365)
    except Exception:
        return df

    # 只有當資料落後 >= 1 天才嘗試 TWSE 補強 (台股 vs 週末也算)
    if (today - last_date).days < 1:
        return df

    # 抓本月 (+ 上個月以防月初)
    fetcher = _fetch_twse_stock_month if ticker.endswith('.TW') else _fetch_tpex_stock_month
    now = datetime.now()
    last_month = (now.replace(day=1) - timedelta(days=1))
    for month_dt in [now, last_month]:
        try:
            supp = fetcher(stock_code, month_dt.strftime('%Y%m'))
            if supp is None or supp.empty:
                continue
            # 對齊 timezone
            if df is not None and not df.empty and df.index.tz is not None:
                supp.index = pd.to_datetime(supp.index).tz_localize(df.index.tz)
            else:
                supp.index = pd.to_datetime(supp.index)
            if df is None or df.empty:
                df = supp
            else:
                # 對齊 timezone 後合併,重複以 TWSE 為準 (更新鮮)
                if df.index.tz is None and supp.index.tz is not None:
                    df.index = pd.to_datetime(df.index).tz_localize(supp.index.tz)
                df = pd.concat([df, supp])
                df = df[~df.index.duplicated(keep='last')].sort_index()
        except Exception:
            continue

    return df


def _fetch_history_best(ticker, period="6mo"):
    """優先用 yf.download (較新、支援明確日期),失敗才回退到 Ticker.history。
    顯式把 end=明天 以確保包含今天已收盤的資料。
    """
    # 把 period 轉成天數
    days_map = {"1mo": 35, "3mo": 100, "6mo": 200, "1y": 400, "2y": 750}
    days = days_map.get(period, 200)
    end = datetime.now() + timedelta(days=1)  # 用 +1 強制包含今天
    start = end - timedelta(days=days)

    df = None
    # 方法 1: yf.download (較常取得今日收盤)
    try:
        df = yf.download(
            ticker,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        # yf.download 2330 個股仍會回 MultiIndex 欄位,壓平
        if df is not None and not df.empty and isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        df = None

    # 方法 2: 退回 Ticker.history
    if df is None or df.empty:
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        except Exception:
            df = None

    df = _clean_history(df)

    # 方法 3: 台股若資料落後超過 1 天,用 TWSE/TPEX 補上最新日
    df = _merge_with_twse(df, ticker)

    return df


def get_stock_data(code, period="6mo"):
    """取得股價資料，並回傳實際使用的 ticker 代號 (.TW 或 .TWO)。"""
    try:
        code = str(code).strip().upper()
        if not code:
            return None, None, None

        # 已含副檔名 (例: 2330.TW, AAPL.US) 或純英文 (美股)
        if '.' in code or not code.replace('-', '').isdigit():
            df = _fetch_history_best(code, period)
            if df is not None and not df.empty:
                return df, yf.Ticker(code), code
            return None, None, None

        # 台股: 先試 .TW 再試 .TWO
        ticker = f"{code}.TW"
        df = _fetch_history_best(ticker, period)
        if df is None or df.empty:
            ticker = f"{code}.TWO"
            df = _fetch_history_best(ticker, period)
        if df is not None and not df.empty:
            return df, yf.Ticker(ticker), ticker
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

def _safe_get(d, *keys, default=None):
    """依序嘗試多個 key,回傳第一個 truthy 值。"""
    for k in keys:
        v = d.get(k) if d else None
        if v is not None and v != 0:
            return v
    return default


@st.cache_data(ttl=21600, show_spinner=False)
def _fetch_twse_bwibbu_day(date_ymd):
    """TWSE 本益比/殖利率/股價淨值比 日報 (所有上市)。"""
    try:
        url = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
        params = {"date": date_ymd, "selectType": "ALL", "response": "json"}
        headers = dict(_HTTP_HEADERS)
        headers["Referer"] = "https://www.twse.com.tw/"
        r = requests.get(url, params=params, timeout=8, headers=headers)
        if r.status_code != 200:
            return None
        j = r.json()
        df = _df_from_any(j)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


@st.cache_data(ttl=21600, show_spinner=False)
def _fetch_tpex_bwibbu_day(date_roc):
    """TPEX 上櫃本益比/殖利率/股價淨值比 日報 (ROC 日期)。"""
    try:
        url = "https://www.tpex.org.tw/www/zh-tw/afterTrading/peRatio"
        params = {"date": date_roc, "id": "", "response": "json"}
        headers = dict(_HTTP_HEADERS)
        headers["Referer"] = "https://www.tpex.org.tw/"
        r = requests.get(url, params=params, timeout=8, headers=headers)
        if r.status_code != 200:
            return None
        j = r.json()
        df = _df_from_any(j)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _fetch_twse_valuation(stock_code, is_otc=False):
    """從 TWSE/TPEX 抓最近 5 個交易日的本益比/P/B/殖利率,回傳第一筆找得到的。"""
    code = str(stock_code).strip()
    for dt in _recent_trading_dates(5):
        if is_otc:
            df = _fetch_tpex_bwibbu_day(_roc_date(dt))
        else:
            df = _fetch_twse_bwibbu_day(dt.strftime('%Y%m%d'))
        if df is None or df.empty:
            continue
        row = _match_stock_row(df, code)
        if row is None:
            continue

        def _f(col_keywords):
            for col in row.index:
                cs = str(col)
                for kw in col_keywords:
                    if kw in cs:
                        v = str(row[col]).replace(',', '').strip()
                        if v in ('', '-', '--', 'N/A'):
                            return None
                        try:
                            return float(v)
                        except Exception:
                            return None
            return None

        return {
            '本益比': _f(['本益比', '_per', 'P/E']),
            '股價淨值比': _f(['股價淨值比', '_pbr', 'P/B']),
            '殖利率': _f(['殖利率', '現金股利殖利率']),
            '日期': dt,
        }
    return None


def _pct_from_ratio(v):
    """yfinance 比例值可能為 0-1 (如 0.28) 或 0-100 (如 28);統一換成 %。"""
    if v is None:
        return 0
    try:
        return v * 100 if abs(v) < 1.5 else v
    except Exception:
        return 0


def get_fundamental_data(ticker_symbol):
    """獲取財務指標。多層來源:
      1. yfinance ticker.info (TTM 指標,美股+台股大部分能用)
      2. TWSE/TPEX BWIBBU_d 日報 (台股官方本益比/P/B/殖利率,幾乎一定有)
      3. 由 P/E + 股價反推 EPS
    只要有 EPS 或 P/E 其中一個就會回傳資料,不再整個失敗。
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        # 先取 info (可能為空)
        eps_info = info.get('trailingEps') or info.get('forwardEps')
        pe_info = info.get('trailingPE')
        pb_info = info.get('priceToBook')
        bvps_info = info.get('bookValue')
        roe_info = info.get('returnOnEquity')
        gm_info = info.get('grossMargins')
        pm_info = info.get('profitMargins')
        dy_info = info.get('dividendYield')
        rg_info = info.get('revenueGrowth')
        price = info.get('currentPrice') or info.get('regularMarketPrice')

        # 對台股補強: 用 TWSE/TPEX 官方日報
        twse_val = None
        is_tw = ticker_symbol.endswith(('.TW', '.TWO'))
        if is_tw:
            stock_code = ticker_symbol.replace('.TW', '').replace('.TWO', '')
            is_otc = ticker_symbol.endswith('.TWO')
            twse_val = _fetch_twse_valuation(stock_code, is_otc=is_otc)
            # 若 info 沒有股價,從最後一根 K 線拿
            if not price:
                try:
                    hist = yf.Ticker(ticker_symbol).history(period='5d')
                    hist = _clean_history(hist)
                    if hist is not None and not hist.empty:
                        price = float(hist.iloc[-1]['Close'])
                except Exception:
                    pass

        # 合併 EPS / PE / PB (以 info 為主,TWSE 為備援)
        pe = pe_info or (twse_val.get('本益比') if twse_val else None)
        pb = pb_info or (twse_val.get('股價淨值比') if twse_val else None)
        dy = dy_info
        if (not dy or dy == 0) and twse_val and twse_val.get('殖利率'):
            dy = twse_val['殖利率']  # TWSE 已經是 % (如 2.5)

        # EPS 反推: 如果 info 沒給但有 PE+price,可以算出 EPS
        eps = eps_info
        if (not eps or eps == 0) and pe and pe > 0 and price:
            eps = price / pe

        # BVPS 反推
        bvps = bvps_info
        if (not bvps or bvps == 0) and pb and pb > 0 and price:
            bvps = price / pb

        # 至少要有 EPS 或 PE 才算有資料
        has_any = any([eps, pe, pb, bvps, roe_info, gm_info, pm_info])
        if not has_any:
            return None

        metrics = {}

        # 1. EPS (已 fallback: info → PE*price 反推)
        metrics['每股盈餘'] = eps or 0

        # 2. 每股淨值 (BVPS) (已 fallback: info → PB*price 反推)
        metrics['每股淨值'] = bvps or 0

        # 3. P/B (先算,後面 ROE 可能用到)
        metrics['股價淨值比'] = pb or 0

        # 4. 股息殖利率 (dy 來源: info 為 0-1 比例,TWSE 為 %)
        if dy is None or dy == 0:
            metrics['股息殖利率'] = 0
        elif dy < 1:
            metrics['股息殖利率'] = dy * 100
        else:
            metrics['股息殖利率'] = dy

        # 5-7. 毛利率 / 淨利率 / 營收 — 試從 info,失敗試 financials
        gm = _pct_from_ratio(gm_info)
        pm = _pct_from_ratio(pm_info)
        rg = _pct_from_ratio(rg_info)
        total_rev = info.get('totalRevenue')

        # financials 補援 (對較小台股有時會有資料)
        try:
            fin = None
            for attr in ('financials', 'quarterly_financials', 'income_stmt', 'quarterly_income_stmt'):
                try:
                    cand = getattr(ticker, attr, None)
                    if cand is not None and not cand.empty:
                        fin = cand
                        break
                except Exception:
                    continue
            if fin is not None:
                latest = fin.iloc[:, 0]
                prev = fin.iloc[:, 1] if fin.shape[1] > 1 else None
                rev = latest.get('Total Revenue') or latest.get('Operating Revenue')
                gp = latest.get('Gross Profit')
                ni = latest.get('Net Income') or latest.get('Net Income Common Stockholders')
                if rev:
                    if not total_rev:
                        total_rev = rev
                    if gm == 0 and gp:
                        gm = gp / rev * 100
                    if pm == 0 and ni:
                        pm = ni / rev * 100
                if rg == 0 and prev is not None:
                    prev_rev = prev.get('Total Revenue') or prev.get('Operating Revenue')
                    if rev and prev_rev:
                        rg = (rev - prev_rev) / abs(prev_rev) * 100
        except Exception:
            pass

        metrics['毛利率'] = gm
        metrics['淨利率'] = pm
        metrics['營收年增率'] = rg

        # 8-9. 總資產週轉率 & 權益乘數
        total_assets = None
        total_equity = None
        for attr in ('balance_sheet', 'quarterly_balance_sheet'):
            try:
                bs = getattr(ticker, attr, None)
                if bs is not None and not bs.empty:
                    col = bs.iloc[:, 0]
                    if total_assets is None:
                        total_assets = col.get('Total Assets')
                    if total_equity is None:
                        total_equity = col.get('Stockholders Equity') or col.get('Total Stockholder Equity')
                    if total_assets and total_equity:
                        break
            except Exception:
                continue
        if not total_equity:
            total_equity = info.get('totalStockholderEquity')

        metrics['總資產週轉率'] = (total_rev / total_assets) if (total_rev and total_assets) else 0
        metrics['權益乘數'] = (total_assets / total_equity) if (total_assets and total_equity) else 0

        # 10. ROE — 優先 info,否則用杜邦恆等式:ROE = EPS / 每股淨值 × 100
        roe = _pct_from_ratio(roe_info)
        if roe == 0 and metrics['每股盈餘'] and metrics['每股淨值']:
            try:
                roe = metrics['每股盈餘'] / metrics['每股淨值'] * 100
            except Exception:
                roe = 0
        metrics['ROE'] = roe

        # 11. 報表期別
        period_ts = info.get('mostRecentQuarter') or info.get('lastFiscalYearEnd')
        if isinstance(period_ts, (int, float)) and period_ts > 0:
            try:
                metrics['報表期別'] = datetime.fromtimestamp(period_ts).strftime('%Y-%m-%d')
            except Exception:
                metrics['報表期別'] = 'TTM'
        elif twse_val:
            metrics['報表期別'] = f"TWSE 日報 ({twse_val['日期'].strftime('%Y-%m-%d')})"
        else:
            metrics['報表期別'] = 'TTM (最近四季)'

        # 12. 資料來源標記 (給 UI 區分)
        metrics['_source'] = 'yfinance+TWSE' if twse_val else 'yfinance'

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
                latest_date = latest.name
                today_date = pd.Timestamp.now(tz=latest_date.tz).normalize() if hasattr(latest_date, 'tz') and latest_date.tz else pd.Timestamp.now().normalize()
                try:
                    is_today = latest_date.date() == today_date.date()
                except Exception:
                    is_today = False
                freshness_label = "今日收盤" if is_today else "最新交易日"

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"💰 {freshness_label}", f"${latest['Close']:.2f}")
                with col2:
                    change = latest['Close'] - df.iloc[-2]['Close']
                    change_pct = (change / df.iloc[-2]['Close']) * 100
                    st.metric("📊 漲跌", f"{change:+.2f}", f"{change_pct:+.2f}%")
                with col3:
                    vol_ratio = latest['Volume'] / latest['Volume_MA5'] if latest['Volume_MA5'] else 0
                    st.metric("📈 量能比", f"{vol_ratio:.2f}x")
                with col4:
                    st.metric("📅 資料日期", latest_date.strftime('%Y-%m-%d'))

                if not is_today:
                    st.caption(
                        f"ℹ️ 今日 ({datetime.now().strftime('%m-%d')}) 收盤資料 Yahoo Finance 尚未更新 — "
                        f"顯示最新完整交易日 ({latest_date.strftime('%m-%d')})。台股通常於 15:00 後更新。"
                    )

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

                if fundamental_data and (
                    fundamental_data.get('每股盈餘', 0) != 0
                    or fundamental_data.get('每股淨值', 0) != 0
                    or fundamental_data.get('股價淨值比', 0) != 0
                ):
                    report_period = fundamental_data.get('報表期別', 'N/A')
                    source = fundamental_data.get('_source', 'yfinance')
                    st.caption(f"📅 財報期別: {report_period} | 代號: {ticker_symbol} | 來源: {source}")

                    # 財務指標
                    st.markdown("##### 💼 財務指標 (最新年度)")

                    def _fmt_pct(v):
                        return f"{v:.2f}%" if v else "N/A"
                    def _fmt_num(v):
                        return f"{v:.2f}" if v else "N/A"

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("毛利率", _fmt_pct(fundamental_data['毛利率']))
                        st.metric("淨利率", _fmt_pct(fundamental_data['淨利率']))

                    with col2:
                        st.metric("總資產週轉率", _fmt_num(fundamental_data['總資產週轉率']))
                        st.metric("權益乘數", _fmt_num(fundamental_data['權益乘數']))

                    with col3:
                        bvps_v = fundamental_data.get('每股淨值', 0)
                        eps_v = fundamental_data.get('每股盈餘', 0)
                        st.metric("每股淨值", f"${bvps_v:.2f}" if bvps_v else "N/A")
                        st.metric("EPS", f"${eps_v:.2f}" if eps_v else "N/A")

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

                    roe_v = fundamental_data.get('ROE', 0)
                    if roe_v:
                        st.info(f"📈 **ROE:** {roe_v:.2f}% (若 yfinance 無提供,由 EPS ÷ 每股淨值 推算)")
                    else:
                        st.info("📈 **ROE:** N/A (無足夠資料推算)")
                    
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
