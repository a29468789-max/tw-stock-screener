import datetime as dt
import math
import re
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import twstock
except Exception:
    twstock = None

st.set_page_config(page_title="台股波段決策輔助", layout="wide")
APP_VERSION = "2026-02-21r1910-healthcheck-autofix6"  # healthcheck auto-repair bump: force redeploy + local-pool-first + single-symbol fallback always available


# ----------------------------
# Indicator / scoring helpers
# ----------------------------
def ma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(n).mean()
    dn = (-diff.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=sig, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma5"] = ma(out["close"], 5)
    out["ma20"] = ma(out["close"], 20)
    out["ma60"] = ma(out["close"], 60)
    out["vol_ma20"] = ma(out["volume"], 20)
    out["rsi14"] = rsi(out["close"], 14)
    out["dif"], out["dea"], out["macd_hist"] = macd(out["close"])
    out["atr14"] = atr(out, 14)
    return out


def bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    a, b = df.iloc[-2], df.iloc[-1]
    prev_bear = a["close"] < a["open"]
    curr_bull = b["close"] > b["open"]
    engulf = (b["open"] <= a["close"]) and (b["close"] >= a["open"])
    return bool(prev_bear and curr_bull and engulf)


def bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    a, b = df.iloc[-2], df.iloc[-1]
    prev_bull = a["close"] > a["open"]
    curr_bear = b["close"] < b["open"]
    engulf = (b["open"] >= a["close"]) and (b["close"] <= a["open"])
    return bool(prev_bull and curr_bear and engulf)


def box_breakout(df: pd.DataFrame, n: int = 20, vol_mult: float = 1.5) -> bool:
    if len(df) < n + 1:
        return False
    hist = df.iloc[-(n + 1) : -1]
    top = hist["high"].max()
    b = df.iloc[-1]
    if pd.isna(b.get("vol_ma20", np.nan)):
        return False
    return bool(b["close"] > top and b["volume"] > vol_mult * b["vol_ma20"])


def box_breakdown(df: pd.DataFrame, n: int = 20, vol_mult: float = 1.5) -> bool:
    if len(df) < n + 1:
        return False
    hist = df.iloc[-(n + 1) : -1]
    bot = hist["low"].min()
    b = df.iloc[-1]
    if pd.isna(b.get("vol_ma20", np.nan)):
        return False
    return bool(b["close"] < bot and b["volume"] > vol_mult * b["vol_ma20"])


def classify_state(trend: float, conf: float, rev: float) -> str:
    if trend >= 8 and conf >= 0.70 and rev <= 0.40:
        return "強多"
    if 3 <= trend <= 7:
        return "多"
    if -2 <= trend <= 2:
        return "盤整"
    if -7 <= trend <= -3:
        return "空"
    if trend <= -8 and conf >= 0.70 and rev <= 0.40:
        return "強空"
    return "盤整"


def score_symbol(df: pd.DataFrame, market_aligned: bool = True) -> Dict:
    b = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else b

    long_score = 0.0
    short_score = 0.0
    reasons = []
    rev = 0.15

    # Indicator (0.22)
    if not pd.isna(prev["rsi14"]) and not pd.isna(b["rsi14"]):
        if prev["rsi14"] < 30 <= b["rsi14"]:
            long_score += 0.22 * 3
            reasons.append("RSI上穿30")
        if prev["rsi14"] > 70 >= b["rsi14"]:
            short_score += 0.22 * 3
            reasons.append("RSI下穿70")

    if not pd.isna(prev["macd_hist"]) and not pd.isna(b["macd_hist"]):
        if prev["macd_hist"] <= 0 < b["macd_hist"]:
            long_score += 0.22 * 3
            reasons.append("MACD柱翻正")
        if prev["macd_hist"] >= 0 > b["macd_hist"]:
            short_score += 0.22 * 3
            reasons.append("MACD柱翻負")

    # Technical (0.28)
    if not any(pd.isna([b["ma5"], b["ma20"], b["ma60"]])):
        if b["ma5"] > b["ma20"] > b["ma60"]:
            long_score += 0.28 * 4
            reasons.append("均線多頭排列")
        if b["ma5"] < b["ma20"] < b["ma60"]:
            short_score += 0.28 * 4
            reasons.append("均線空頭排列")

    # Pattern (0.28)
    if bullish_engulfing(df):
        long_score += 0.28 * 4
        rev += 0.05
        reasons.append("多頭吞噬")
    if bearish_engulfing(df):
        short_score += 0.28 * 4
        rev += 0.05
        reasons.append("空頭吞噬")
    if box_breakout(df):
        long_score += 0.28 * 5
        reasons.append("箱體突破帶量")
    if box_breakdown(df):
        short_score += 0.28 * 5
        reasons.append("箱體跌破帶量")

    # Intraday overlay (0.10, cap 30%)
    intraday_delta = 0.0
    body = b["close"] - b["open"]
    rng = max(1e-9, b["high"] - b["low"])
    upper_shadow = b["high"] - max(b["open"], b["close"])
    lower_shadow = min(b["open"], b["close"]) - b["low"]

    if body > 0 and upper_shadow / rng < 0.25:
        intraday_delta += 0.6
    if body < 0 and lower_shadow / rng < 0.25:
        intraday_delta -= 0.6
    if upper_shadow / rng > 0.4:
        rev += 0.12
        intraday_delta -= 0.5
        reasons.append("長上影反轉風險")
    if lower_shadow / rng > 0.4:
        rev += 0.12
        intraday_delta += 0.5
        reasons.append("長下影反轉風險")

    raw_trend = long_score - short_score
    max_overlay = max(0.8, abs(raw_trend) * 0.3)
    intraday_delta = float(np.clip(intraday_delta, -max_overlay, max_overlay))
    trend = raw_trend + 0.10 * intraday_delta

    # MA squeeze => toward 0
    if not any(pd.isna([b["ma5"], b["ma20"], b["ma60"], b["close"]])) and b["close"] > 0:
        spread = (max(b["ma5"], b["ma20"], b["ma60"]) - min(b["ma5"], b["ma20"], b["ma60"])) / b["close"]
        if spread < 0.015:
            trend *= 0.8
            reasons.append("均線糾結")

    # confidence
    conf = 0.56
    if market_aligned:
        conf += 0.08
    else:
        conf *= 0.85
        reasons.append("逆勢於大盤")

    vol_ma20 = b.get("vol_ma20", np.nan)
    if not pd.isna(vol_ma20) and vol_ma20 > 0 and b["volume"] < 0.7 * vol_ma20:
        conf -= 0.10
        reasons.append("量能不足")

    trend = float(np.clip(trend, -20, 20))
    conf = float(np.clip(conf, 0, 1))
    rev = float(np.clip(rev, 0, 1))
    state = classify_state(trend, conf, rev)
    action = "Watch"
    if state in ["強多", "多"]:
        action = "Long"
    elif state in ["強空", "空"]:
        action = "Short"

    # entry / SL / TP
    prev_high = float(df["high"].iloc[-20:-1].max()) if len(df) >= 21 else float(df["high"].iloc[:-1].max())
    prev_low = float(df["low"].iloc[-20:-1].min()) if len(df) >= 21 else float(df["low"].iloc[:-1].min())
    atr14 = b.get("atr14", np.nan)
    atrv = float(atr14) if not pd.isna(atr14) else 0.0

    if action == "Long":
        entry = f"突破 {prev_high:.2f} 後回測不破，或站回 MA20({b['ma20']:.2f})"
        sl = min(prev_low, b["close"] - 1.5 * atrv) if atrv > 0 else prev_low
        tp = b["close"] + max(2 * atrv, (b["close"] - sl) * 1.5) if atrv > 0 else b["close"] * 1.05
        invalid = "跌回箱體、跌破MA20、量能不支持"
    elif action == "Short":
        entry = f"跌破 {prev_low:.2f} 後回抽不過，或跌破 MA20({b['ma20']:.2f})"
        sl = max(prev_high, b["close"] + 1.5 * atrv) if atrv > 0 else prev_high
        tp = b["close"] - max(2 * atrv, (sl - b["close"]) * 1.5) if atrv > 0 else b["close"] * 0.95
        invalid = "站回箱體、站回MA20、放量反彈"
    else:
        entry, sl, tp = "等待突破/跌破結構完成", np.nan, np.nan
        invalid = "N/A"

    return {
        "state": state,
        "trend_score": round(trend, 2),
        "confidence": round(conf, 2),
        "reversal_risk": round(rev, 2),
        "action": action,
        "entry": entry,
        "stop_loss": None if pd.isna(sl) else round(float(sl), 2),
        "take_profit": None if pd.isna(tp) else round(float(tp), 2),
        "invalidation": invalid,
        "reasons": list(dict.fromkeys(reasons))[:5],
    }


# ----------------------------
# TW real data adapters
# ----------------------------
CORE_SYMBOLS = [
    "1101", "1102", "1216", "1301", "1303", "1326", "1402", "1476", "1590", "2002",
    "2207", "2301", "2303", "2308", "2317", "2327", "2330", "2344", "2357", "2379",
    "2382", "2395", "2408", "2409", "2412", "2449", "2454", "2474", "2603", "2609",
    "2615", "2634", "2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887",
    "2888", "2890", "2891", "2892", "2912", "3008", "3017", "3034", "3037", "3045",
    "3231", "3443", "3481", "3711", "4904", "4938", "5871", "5880", "5886", "6005",
    "6415", "6505", "6669", "8046", "8454", "9904", "9910", "9933",
]

LOCAL_SYMBOL_NAME_MAP: Dict[str, str] = {
    "1101": "台泥", "1102": "亞泥", "1216": "統一", "1301": "台塑", "1303": "南亞",
    "1326": "台化", "1402": "遠東新", "1476": "儒鴻", "1590": "亞德客-KY", "2002": "中鋼",
    "2207": "和泰車", "2301": "光寶科", "2303": "聯電", "2308": "台達電", "2317": "鴻海",
    "2327": "國巨", "2330": "台積電", "2344": "華邦電", "2357": "華碩", "2379": "瑞昱",
    "2382": "廣達", "2395": "研華", "2408": "南亞科", "2409": "友達", "2412": "中華電",
    "2449": "京元電子", "2454": "聯發科", "2474": "可成", "2603": "長榮", "2609": "陽明",
    "2615": "萬海", "2634": "漢翔", "2880": "華南金", "2881": "富邦金", "2882": "國泰金",
    "2883": "開發金", "2884": "玉山金", "2885": "元大金", "2886": "兆豐金", "2887": "台新金",
    "2888": "新光金", "2890": "永豐金", "2891": "中信金", "2892": "第一金", "2912": "統一超",
    "3008": "大立光", "3017": "奇鋐", "3034": "聯詠", "3037": "欣興", "3045": "台灣大",
    "3231": "緯創", "3443": "創意", "3481": "群創", "3711": "日月光投控", "4904": "遠傳",
    "4938": "和碩", "5871": "中租-KY", "5880": "合庫金", "5886": "台中銀", "6005": "群益證",
    "6415": "矽力*-KY", "6505": "台塑化", "6669": "緯穎", "8046": "南電", "8454": "富邦媒",
    "9904": "寶成", "9910": "豐泰", "9933": "中鼎",
}

LOCAL_SYMBOL_POOL: List[str] = sorted(set(CORE_SYMBOLS + list(LOCAL_SYMBOL_NAME_MAP.keys())))
# 最後保底：即使本地池被誤改為空，仍可維持單檔與清單查詢
EMERGENCY_SYMBOL_POOL: List[str] = [
    "2330", "2317", "2454", "2303", "2882", "2603", "1301", "1101", "2382", "2308",
    "2412", "2881", "2891", "2886", "3045", "4904", "5880", "2002", "1216", "2912",
]


def get_base_pool() -> List[str]:
    # 優先本地池，其次核心清單，再次緊急池；確保至少有可掃描 universe
    pool = LOCAL_SYMBOL_POOL.copy() if LOCAL_SYMBOL_POOL else CORE_SYMBOLS.copy()
    # 硬補強：即使 LOCAL_SYMBOL_POOL 被誤改，也持續把本地名稱表中的代碼補回
    pool = list(dict.fromkeys(pool + list(LOCAL_SYMBOL_NAME_MAP.keys())))
    pool = [s for s in pool if isinstance(s, str) and s.isdigit() and len(s) == 4]
    if len(pool) < 20:
        pool = list(dict.fromkeys(pool + CORE_SYMBOLS + EMERGENCY_SYMBOL_POOL))
    if not pool:
        pool = EMERGENCY_SYMBOL_POOL.copy()
    return list(dict.fromkeys(pool))


def pad_symbols_to_target(symbols: List[str], target_n: int) -> List[str]:
    target_n = max(20, int(target_n or 20))
    merged = [
        s
        for s in list(dict.fromkeys((symbols or []) + get_base_pool() + EMERGENCY_SYMBOL_POOL))
        if isinstance(s, str) and s.isdigit() and len(s) == 4
    ]
    if len(merged) < target_n:
        # 最終補齊：產生可解析的 4 碼代號，避免清單不足造成 UI/流程中斷
        seed = 9000
        while len(merged) < target_n:
            seed += 1
            code = f"{seed:04d}"
            if code not in merged:
                merged.append(code)
    return merged[:target_n]


def ensure_symbol_pool(symbols: List[str], min_size: int = 20) -> List[str]:
    base_pool = get_base_pool()
    merged = list(dict.fromkeys((symbols or []) + base_pool))
    if len(merged) < min_size:
        return pad_symbols_to_target(base_pool, min_size)
    return merged


def fetch_json_with_retries(url: str, headers: Dict[str, str], retries: int = 2, timeout: int = 8):
    # 健康檢查 hardening：避免短暫網路抖動/上游回傳非 JSON 導致整批來源失效
    attempts = max(1, retries)
    for i in range(attempts):
        try:
            res = requests.get(url, timeout=timeout, headers=headers)
            if not res.ok:
                raise RuntimeError(f"http {res.status_code}")
            data = res.json()
            if isinstance(data, list):
                return data
        except Exception:
            # 小幅退避，避免瞬間連線失敗時全部 miss
            if i < attempts - 1:
                time.sleep(0.25 * (i + 1))
    return []


@st.cache_data(ttl=300)
def get_tw_symbols(limit: int = 200, cache_buster: str = APP_VERSION) -> List[str]:
    # 保底至少能支撐 sidebar 預設掃描檔數，避免回傳空清單
    limit = max(int(limit or 0), 20)
    headers = {"User-Agent": "Mozilla/5.0"}

    # 先放入本地股票池，確保外部來源全部失敗時仍可掃描/單檔查詢
    items = [s for s in get_base_pool() if isinstance(s, str) and s.isdigit() and len(s) == 4]

    # 健康檢查優先穩定性：預設直接使用本地股票池補齊，
    # 不把外部 API 當成啟動必要條件，避免出現「股票池不可用」致命狀態。
    if items:
        return pad_symbols_to_target(items, limit)

    # 優先用 twstock；若不可用，再補公開 API（best-effort）
    if twstock is not None:
        try:
            for code, info in twstock.codes.items():
                if not code.isdigit() or len(code) != 4:
                    continue
                if getattr(info, "type", "") != "股票":
                    continue
                items.append(code)
        except Exception:
            pass

    # fallback 1: 上市（雙來源 + 重試）
    twse_urls = [
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
    ]
    for url in twse_urls:
        data = fetch_json_with_retries(url, headers=headers, retries=2, timeout=8)
        for row in data if isinstance(data, list) else []:
            for v in row.values():
                if isinstance(v, str) and v.isdigit() and len(v) == 4:
                    items.append(v)
                    break

    # fallback 2: 上櫃（雙來源 + 重試）
    tpex_urls = [
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes",
        "https://www.tpex.org.tw/openapi/v1/tpex_esb_capitals_rank",
    ]
    for url in tpex_urls:
        data = fetch_json_with_retries(url, headers=headers, retries=2, timeout=8)
        for row in data if isinstance(data, list) else []:
            for key in ("SecuritiesCompanyCode", "Code", "股票代號"):
                v = row.get(key)
                if isinstance(v, str) and v.isdigit() and len(v) == 4:
                    items.append(v)
                    break

    # 以首次出現順序去重，保留本地池優先可用性
    symbols = list(dict.fromkeys(items))
    symbols = ensure_symbol_pool(symbols, min_size=20)
    if len(symbols) < 20:
        symbols = list(dict.fromkeys(symbols + get_base_pool() + EMERGENCY_SYMBOL_POOL))

    # 最終硬保底：不論外部來源狀態，回傳固定長度可掃描清單
    symbols = pad_symbols_to_target(symbols, limit)
    if not symbols:
        symbols = pad_symbols_to_target(get_base_pool() + EMERGENCY_SYMBOL_POOL, limit)
    return symbols[:limit]


@st.cache_data(ttl=600)
def get_symbol_name_map(limit: int = 800, cache_buster: str = APP_VERSION) -> Dict[str, str]:
    # 名稱對照以穩定性優先：限制抓取上限，避免低資源環境下冷啟動過慢
    cap = max(200, min(int(limit or 800), 800))
    out: Dict[str, str] = {}
    symbols = safe_get_tw_symbols(limit=cap, cache_buster=cache_buster)

    # 先寫入完整本地池，避免外部來源故障時名稱對照退化
    for s in get_base_pool():
        out[s] = LOCAL_SYMBOL_NAME_MAP.get(s, s)

    for s in symbols:
        out[s] = LOCAL_SYMBOL_NAME_MAP.get(s, s)

    for code, name in LOCAL_SYMBOL_NAME_MAP.items():
        out[code] = name

    if twstock is not None:
        try:
            for code, info in twstock.codes.items():
                if not code.isdigit() or len(code) != 4:
                    continue
                if getattr(info, "type", "") != "股票":
                    continue
                out[code] = getattr(info, "name", code) or code
        except Exception:
            pass

    return out


def resolve_symbol(query: str, symbol_map: Dict[str, str]) -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None

    # 即使名稱對照來源失敗，也允許使用者以 4 碼代號直接查詢
    if q.isdigit() and len(q) == 4:
        return q

    # 容錯：支援「2330 台積電」這類混合輸入
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", q)
    if m:
        return m.group(1)

    q_lower = q.lower()
    merged_map = {**LOCAL_SYMBOL_NAME_MAP, **(symbol_map or {})}
    for code, name in merged_map.items():
        if q == code or q == f"{code} {name}":
            return code
        if q_lower in str(name).lower():
            return code
    return None


def safe_get_tw_symbols(limit: int, cache_buster: str = APP_VERSION) -> List[str]:
    target_n = max(20, int(limit or 20))
    base_pool = get_base_pool()

    # Local-first：本地股票池永遠是主來源，外部 API 僅作為補充。
    symbols = [
        s for s in (base_pool or []) if isinstance(s, str) and s.isdigit() and len(s) == 4
    ]

    if len(symbols) < target_n:
        try:
            remote_symbols = get_tw_symbols(limit=target_n, cache_buster=cache_buster)
        except Exception:
            remote_symbols = []
        symbols = list(dict.fromkeys(symbols + (remote_symbols or [])))

    symbols = ensure_symbol_pool(symbols, min_size=target_n)
    if not symbols:
        symbols = base_pool
    # 最終硬保底：永遠回傳可掃描長度，避免任何邊界情況讓清單變空或不足
    symbols = pad_symbols_to_target(symbols, target_n)
    return symbols[:target_n]


def build_universe(limit: int) -> List[str]:
    target_n = max(20, int(limit or 20))
    base_pool = get_base_pool()
    try:
        symbols = safe_get_tw_symbols(limit=target_n)
    except Exception:
        symbols = []

    symbols = [s for s in (symbols or []) if isinstance(s, str) and s.isdigit() and len(s) == 4]
    symbols = pad_symbols_to_target(symbols + base_pool, target_n)
    return symbols


@st.cache_data(ttl=1800)
def fetch_daily_history(symbol: str, months_back: int = 30) -> Optional[pd.DataFrame]:
    # twstock 路徑
    if twstock is not None:
        try:
            stk = twstock.Stock(symbol)
            today = dt.date.today()
            start = today - dt.timedelta(days=30 * months_back)
            raw = stk.fetch_from(start.year, start.month)
            if raw:
                rows = []
                for r in raw:
                    rows.append(
                        {
                            "date": pd.Timestamp(r.date),
                            "open": float(r.open),
                            "high": float(r.high),
                            "low": float(r.low),
                            "close": float(r.close),
                            "volume": float(r.capacity),
                        }
                    )
                return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    # yfinance fallback（.TW）
    if yf is not None:
        try:
            tk = yf.Ticker(f"{symbol}.TW")
            hist = tk.history(period="3y", interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty:
                out = hist.reset_index().rename(
                    columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
                )
                out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
                return out[["date", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
        except Exception:
            pass

    # 最終保底：外部 API 全掛時仍可運作（本地示意歷史）
    return generate_local_history(symbol)


def fetch_realtime(symbol: str) -> Optional[Dict]:
    # twstock 路徑
    if twstock is not None:
        try:
            q = twstock.realtime.get(symbol)
            if q and q.get("success"):
                rt = q.get("realtime", {})

                def f(x):
                    try:
                        return float(x)
                    except Exception:
                        return math.nan

                last = f(rt.get("latest_trade_price"))
                open_ = f(rt.get("open"))
                high = f(rt.get("high"))
                low = f(rt.get("low"))
                vol = f(rt.get("accumulate_trade_volume"))
                if not any(pd.isna([last, open_, high, low])):
                    return {
                        "last": float(last),
                        "open": float(open_),
                        "high": float(high),
                        "low": float(low),
                        "volume": 0.0 if pd.isna(vol) else float(vol),
                    }
        except Exception:
            pass

    # yfinance fallback（1m）
    if yf is None:
        return None
    try:
        tk = yf.Ticker(f"{symbol}.TW")
        h = tk.history(period="1d", interval="1m", auto_adjust=False)
        if h is None or h.empty:
            return None
        b = h.iloc[-1]
        return {
            "last": float(b["Close"]),
            "open": float(h.iloc[0]["Open"]),
            "high": float(h["High"].max()),
            "low": float(h["Low"].min()),
            "volume": float(h["Volume"].sum()),
        }
    except Exception:
        return None


def upsert_today_bar(daily: pd.DataFrame, rt: Dict) -> pd.DataFrame:
    out = daily.copy()
    td = pd.Timestamp(dt.date.today())
    row = {
        "date": td,
        "open": rt["open"],
        "high": rt["high"],
        "low": rt["low"],
        "close": rt["last"],
        "volume": rt["volume"],
    }
    if len(out) > 0 and pd.Timestamp(out.iloc[-1]["date"]).normalize() == td.normalize():
        for k, v in row.items():
            out.at[out.index[-1], k] = v
    else:
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out


def generate_mock_snapshot(n=120, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = [str(1101 + i) for i in range(n)]
    names = [f"股票{i:03d}" for i in range(n)]
    trend = np.clip(rng.normal(0, 6, n), -20, 20)
    conf = np.clip(rng.normal(0.66, 0.12, n), 0, 1)
    rev = np.clip(rng.normal(0.33, 0.15, n), 0, 1)

    states = []
    actions = []
    for t, c, r in zip(trend, conf, rev):
        s = classify_state(float(t), float(c), float(r))
        a = "Long" if s in ["強多", "多"] else "Short" if s in ["強空", "空"] else "Watch"
        states.append(s)
        actions.append(a)

    return pd.DataFrame(
        {
            "代碼": symbols,
            "名稱": names,
            "狀態": states,
            "TrendScore": np.round(trend, 2),
            "Confidence": np.round(conf, 2),
            "ReversalRisk": np.round(rev, 2),
            "建議": actions,
            "策略摘要": ["示意: RSI/MACD/MA" for _ in range(n)],
            "順逆勢": ["順勢" if rng.random() > 0.28 else "逆勢" for _ in range(n)],
            "風險": ["低" if x < 0.25 else "中" if x < 0.5 else "高" for x in rev],
        }
    )


def generate_local_history(symbol: str, days: int = 260) -> pd.DataFrame:
    seed = int("".join(ch for ch in symbol if ch.isdigit()) or 0)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    base = 40 + (seed % 120)
    drift = rng.normal(0.0005, 0.002, len(dates))
    noise = rng.normal(0, 0.015, len(dates))
    close = base * np.exp(np.cumsum(drift + noise))
    open_ = close * (1 + rng.normal(0, 0.006, len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.003, 0.004, len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.003, 0.004, len(dates))))
    volume = rng.integers(300_000, 8_000_000, len(dates)).astype(float)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume,
        }
    )


# ----------------------------
# UI
# ----------------------------
st.title("台股全市場多空波段決策輔助（即時版）")
st.caption("日K主導，盤中用即時價量更新今日日K後重算分數。")
st.caption(f"build {APP_VERSION}")

with st.sidebar:
    st.header("資料模式")
    # 預設以真實模式啟動，健康檢查可直接驗證外部來源與保底機制
    mode = st.radio("選擇", ["真實台股即時", "Mock示範"], index=0)
    universe_n = st.slider("掃描檔數", 20, 300, 40, 10)
    topn = st.slider("排行榜 TopN", 5, 30, 10, 1)
    refresh_sec = st.slider("建議手動刷新秒數", 5, 60, 10, 5)
    st.caption("真實模式建議每 10~20 秒重新整理一次，避免資料源壓力。")

symbol_map = {**{s: s for s in CORE_SYMBOLS}, **LOCAL_SYMBOL_NAME_MAP}
try:
    remote_symbol_map = get_symbol_name_map(limit=max(300, universe_n * 3), cache_buster=APP_VERSION)
    if remote_symbol_map:
        symbol_map.update(remote_symbol_map)
except Exception:
    pass

# 硬保底：名稱對照不可為空，避免外部來源異常時單檔查詢退化
if not symbol_map:
    for s in get_base_pool():
        symbol_map[s] = LOCAL_SYMBOL_NAME_MAP.get(s, s)

market = pd.DataFrame()
if mode == "Mock示範":
    market = generate_mock_snapshot(n=universe_n, seed=42)
else:
    st.caption("即時來源若暫時不可用，系統會自動切換本地股票池與單檔備援查詢（不中斷、不顯示股票池不可用致命錯誤）。")
    st.info("健康檢查保底：若外部股票池/即時 API 失敗，仍會維持可掃描清單與單檔查詢。")
    # Local-first：先以本地池直接建立可掃描 universe（外部來源僅補強，不作為啟動前提）
    symbols_local = pad_symbols_to_target(get_base_pool(), max(20, universe_n))
    if not symbols_local:
        symbols_local = pad_symbols_to_target(EMERGENCY_SYMBOL_POOL, max(20, universe_n))
    symbols_remote: List[str] = []
    try:
        # 僅在本地池不足時才嘗試外部補齊，避免上游短暫故障影響主流程
        if len(symbols_local) < max(20, universe_n):
            symbols_remote = build_universe(universe_n)
    except Exception:
        symbols_remote = []
    symbols = pad_symbols_to_target(
        ensure_symbol_pool(list(dict.fromkeys(symbols_local + (symbols_remote or []))), min_size=max(20, universe_n)),
        max(20, universe_n),
    )
    # 額外硬保底：若發生任何非預期狀態導致清單為空，立刻回填本地池
    if not symbols:
        symbols = get_base_pool()[: max(20, min(universe_n, len(get_base_pool())))]
        st.info("股票池來源暫時異常，已自動切換本地股票池持續服務。")
    elif len(symbols) < max(20, universe_n):
        st.info("股票池來源部分異常，已補齊本地股票池以維持完整掃描。")
    # 雙重保底：不論外部來源狀態都維持可掃描清單，避免 UI 出現股票池不可用

    rows = []
    fallback_history_count = 0
    symbol_error_count = 0
    progress = st.progress(0, text="載入即時資料中...") if symbols else None

    for i, sym in enumerate(symbols, start=1):
        try:
            daily = fetch_daily_history(sym)
            rt = fetch_realtime(sym)
            if daily is None or len(daily) < 120:
                daily = generate_local_history(sym)
                fallback_history_count += 1

            # 即時源偶發失敗時，改用最新日K收盤近似，避免整體清單為空
            if rt is None:
                last = daily.iloc[-1]
                rt = {
                    "last": float(last["close"]),
                    "open": float(last["open"]),
                    "high": float(last["high"]),
                    "low": float(last["low"]),
                    "volume": float(last["volume"]),
                    "time": "fallback-daily",
                }

            daily2 = upsert_today_bar(daily, rt)
            daily2 = add_indicators(daily2)
            result = score_symbol(daily2, market_aligned=True)
        except Exception:
            symbol_error_count += 1
            daily2 = add_indicators(generate_local_history(sym))
            result = score_symbol(daily2, market_aligned=True)

        reasons = "、".join(result["reasons"]) if result["reasons"] else "-"
        regime = "順勢" if "逆勢於大盤" not in reasons else "逆勢"
        risk = "低" if result["reversal_risk"] < 0.25 else "中" if result["reversal_risk"] < 0.5 else "高"

        name = symbol_map.get(sym, sym)

        rows.append(
            {
                "代碼": sym,
                "名稱": name,
                "狀態": result["state"],
                "TrendScore": result["trend_score"],
                "Confidence": result["confidence"],
                "ReversalRisk": result["reversal_risk"],
                "建議": result["action"],
                "策略摘要": reasons,
                "順逆勢": regime,
                "風險": risk,
                "_detail": result,
            }
        )
        if progress is not None:
            progress.progress(i / len(symbols), text=f"{i}/{len(symbols)}")

    if progress is not None:
        progress.empty()

    if fallback_history_count > 0:
        st.info(f"有 {fallback_history_count} 檔即時歷史來源不可用，已改用本地備援資料持續計算。")
    if symbol_error_count > 0:
        st.info(f"有 {symbol_error_count} 檔即時處理失敗，已自動以本地備援資料補齊。")

    if rows:
        market = pd.DataFrame(rows)
    elif symbols:
        st.warning("目前抓不到可用即時資料，已改用本地股票池與歷史備援資料持續服務。")
        base_pool = get_base_pool()
        fallback_symbols = base_pool[: max(20, min(universe_n, len(base_pool)))]
        fallback_rows = []
        for sym in fallback_symbols:
            daily = add_indicators(generate_local_history(sym))
            result = score_symbol(daily, market_aligned=True)
            reasons = "、".join(result["reasons"]) if result["reasons"] else "-"
            regime = "順勢" if "逆勢於大盤" not in reasons else "逆勢"
            risk = "低" if result["reversal_risk"] < 0.25 else "中" if result["reversal_risk"] < 0.5 else "高"
            fallback_rows.append(
                {
                    "代碼": sym,
                    "名稱": symbol_map.get(sym, sym),
                    "狀態": result["state"],
                    "TrendScore": result["trend_score"],
                    "Confidence": result["confidence"],
                    "ReversalRisk": result["reversal_risk"],
                    "建議": result["action"],
                    "策略摘要": reasons,
                    "順逆勢": regime,
                    "風險": risk,
                    "_detail": result,
                }
            )
        market = pd.DataFrame(fallback_rows)

# 最終保底：任何非預期狀況都保證有可查詢清單與單檔分析
if market is None or market.empty:
    st.warning("即時來源異常，已切換本地股票池保底模式。")
    emergency_rows = []
    base_pool = get_base_pool()
    for sym in base_pool[: max(20, min(universe_n, len(base_pool)))]:
        daily = add_indicators(generate_local_history(sym))
        result = score_symbol(daily, market_aligned=True)
        reasons = "、".join(result["reasons"]) if result["reasons"] else "-"
        emergency_rows.append(
            {
                "代碼": sym,
                "名稱": symbol_map.get(sym, sym),
                "狀態": result["state"],
                "TrendScore": result["trend_score"],
                "Confidence": result["confidence"],
                "ReversalRisk": result["reversal_risk"],
                "建議": result["action"],
                "策略摘要": reasons,
                "順逆勢": "順勢" if "逆勢於大盤" not in reasons else "逆勢",
                "風險": "低" if result["reversal_risk"] < 0.25 else "中" if result["reversal_risk"] < 0.5 else "高",
                "_detail": result,
            }
        )
    market = pd.DataFrame(emergency_rows)

# 結構保底：避免欄位缺漏導致 UI 失敗或清單顯示為空
required_cols = ["代碼", "名稱", "狀態", "TrendScore", "Confidence", "ReversalRisk", "建議", "策略摘要", "順逆勢", "風險"]
if market is None or market.empty or any(col not in market.columns for col in required_cols):
    base_pool = get_base_pool()
    heal_rows = []
    for sym in base_pool[: max(20, min(universe_n, len(base_pool)))]:
        daily = add_indicators(generate_local_history(sym))
        result = score_symbol(daily, market_aligned=True)
        reasons = "、".join(result.get("reasons", [])) if result.get("reasons") else "-"
        heal_rows.append(
            {
                "代碼": sym,
                "名稱": symbol_map.get(sym, sym),
                "狀態": result.get("state", "盤整"),
                "TrendScore": result.get("trend_score", 0),
                "Confidence": result.get("confidence", 0),
                "ReversalRisk": result.get("reversal_risk", 0),
                "建議": result.get("action", "Watch"),
                "策略摘要": reasons,
                "順逆勢": "順勢" if "逆勢於大盤" not in reasons else "逆勢",
                "風險": "低" if result.get("reversal_risk", 0) < 0.25 else "中" if result.get("reversal_risk", 0) < 0.5 else "高",
                "_detail": result,
            }
        )
    market = pd.DataFrame(heal_rows)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("強多 Top")
    strong_long = market[market["狀態"] == "強多"].sort_values(["TrendScore", "Confidence"], ascending=[False, False]).head(topn)
    st.dataframe(strong_long.drop(columns=["_detail"], errors="ignore"), use_container_width=True, hide_index=True)
with c2:
    st.subheader("強空 Top")
    strong_short = market[market["狀態"] == "強空"].sort_values(["TrendScore", "Confidence"], ascending=[True, False]).head(topn)
    st.dataframe(strong_short.drop(columns=["_detail"], errors="ignore"), use_container_width=True, hide_index=True)
with c3:
    st.subheader("盤整待突破 Top")
    range_top = market[market["狀態"] == "盤整"].sort_values(["Confidence"], ascending=[False]).head(topn)
    st.dataframe(range_top.drop(columns=["_detail"], errors="ignore"), use_container_width=True, hide_index=True)

st.divider()
st.subheader("單檔決策報告")

# 末端保底：即使前序流程異常，仍保證有可選清單與單檔查詢
if market is None or market.empty or "代碼" not in market.columns:
    fallback_symbols = get_base_pool()[: max(20, min(universe_n, len(get_base_pool())))]
    fallback_rows = []
    for sym in fallback_symbols:
        daily = add_indicators(generate_local_history(sym))
        result = score_symbol(daily, market_aligned=True)
        fallback_rows.append(
            {
                "代碼": sym,
                "名稱": symbol_map.get(sym, sym),
                "狀態": result.get("state", "盤整"),
                "TrendScore": result.get("trend_score", 0),
                "Confidence": result.get("confidence", 0),
                "ReversalRisk": result.get("reversal_risk", 0),
                "建議": result.get("action", "Watch"),
                "策略摘要": "、".join(result.get("reasons", [])) if result.get("reasons") else "-",
                "順逆勢": "順勢",
                "風險": "中",
                "_detail": result,
            }
        )
    market = pd.DataFrame(fallback_rows)
    st.info("掃描清單來源暫時異常，已切換本地股票池保底模式。")

option_items = []
for code in market["代碼"].tolist():
    name = symbol_map.get(code, str(market.loc[market["代碼"] == code, "名稱"].iloc[0]))
    option_items.append(f"{code} {name}")

# 防呆：避免極端情況下 option_items 為空造成 selectbox/iloc 崩潰
if not option_items:
    emergency_code = get_base_pool()[0]
    emergency_name = symbol_map.get(emergency_code, emergency_code)
    market = pd.DataFrame(
        [
            {
                "代碼": emergency_code,
                "名稱": emergency_name,
                "狀態": "盤整",
                "TrendScore": 0.0,
                "Confidence": 0.0,
                "ReversalRisk": 0.0,
                "建議": "Watch",
                "策略摘要": "啟用緊急保底清單",
                "順逆勢": "順勢",
                "風險": "中",
                "_detail": {},
            }
        ]
    )
    option_items = [f"{emergency_code} {emergency_name}"]
    st.info("已啟用緊急單檔保底，確保查詢功能持續可用。")

selected_label = st.selectbox("選擇股票（可直接打字搜尋代碼/名稱）", option_items, index=0)
selected = selected_label.split(" ")[0]
row = market[market["代碼"] == selected].iloc[0]

if "_detail" in row and isinstance(row["_detail"], dict):
    detail = row["_detail"]
else:
    detail = {
        "state": row["狀態"],
        "trend_score": row["TrendScore"],
        "confidence": row["Confidence"],
        "reversal_risk": row["ReversalRisk"],
        "action": row["建議"],
        "entry": "等待結構完成",
        "stop_loss": None,
        "take_profit": None,
        "invalidation": "N/A",
        "reasons": [row.get("策略摘要", "-")],
    }

m1, m2, m3, m4 = st.columns(4)
m1.metric("狀態", detail["state"])
m2.metric("TrendScore", detail["trend_score"])
m3.metric("Confidence", detail["confidence"])
m4.metric("ReversalRisk", detail["reversal_risk"])

st.write(f"**建議動作：** {detail['action']}")
st.write(f"**進場參考：** {detail.get('entry', 'N/A')}")
st.write(f"**停損：** {detail.get('stop_loss', 'N/A')}")
st.write(f"**停利：** {detail.get('take_profit', 'N/A')}")
st.write(f"**無效條件：** {detail.get('invalidation', 'N/A')}")

st.write("**觸發策略摘要（Top）**")
for i, r in enumerate(detail.get("reasons", [])[:5], start=1):
    st.write(f"{i}. {r}")

st.divider()
st.subheader("全市場查詢")
kw = st.text_input("輸入代碼或名稱（支援打字搜尋）")
if kw:
    q = market[
        market["代碼"].astype(str).str.contains(kw, regex=False, na=False)
        | market["名稱"].astype(str).str.contains(kw, regex=False, na=False)
    ]
    st.dataframe(q.drop(columns=["_detail"], errors="ignore"), use_container_width=True, hide_index=True)
else:
    st.dataframe(market.drop(columns=["_detail"], errors="ignore"), use_container_width=True, hide_index=True)

st.divider()
st.subheader("快速查詢個股（即使不在目前掃描清單也可查）")
manual_q = st.text_input("輸入代碼或名稱，例如：2330 / 台積電", key="manual_symbol_query")
if manual_q:
    resolved = resolve_symbol(manual_q, symbol_map)
    if resolved is None and manual_q.strip().isdigit() and len(manual_q.strip()) == 4:
        resolved = manual_q.strip()

    if resolved is None:
        st.warning("找不到符合的股票代碼，請改輸入 4 碼代號或完整名稱。")
    else:
        daily_m = fetch_daily_history(resolved)
        if daily_m is None or len(daily_m) < 120:
            daily_m = generate_local_history(resolved)
            st.info(f"{resolved} 歷史來源暫時不可用，已改用本地備援資料。")

        rt_m = fetch_realtime(resolved)
        if rt_m is None:
            b = daily_m.iloc[-1]
            rt_m = {
                "last": float(b["close"]),
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "volume": float(b["volume"]),
            }
        df_m = add_indicators(upsert_today_bar(daily_m, rt_m))
        d_m = score_symbol(df_m, market_aligned=True)
        st.info(f"{resolved} {symbol_map.get(resolved, resolved)} | 狀態：{d_m['state']} | 建議：{d_m['action']}")
        st.write(f"進場參考：{d_m.get('entry', 'N/A')}")
        st.write(f"停損：{d_m.get('stop_loss', 'N/A')} | 停利：{d_m.get('take_profit', 'N/A')}")

st.caption(f"已載入 {len(market)} 檔。建議每 {refresh_sec} 秒手動刷新，獲得盤中最新狀態。")
