"""
市场数据获取脚本 v2.0
输出 JSON 格式供 Claude 分析使用
自动从 Config/Holdings.md 读取持仓配置

功能模块：
1. 行情数据：指数、ETF、A股、港股
2. 宏观数据：PMI、CPI、社融等
3. 资金数据：北向资金、板块资金流
4. 公告数据：持仓标的最新公告
5. 快讯数据：财联社电报（替代WebSearch）
6. 技术指标：MA/RSI/MACD/区间位置
"""

import akshare as ak
import pandas as pd
import json
import re
import os
import sys
from datetime import datetime, timedelta

# ============ 配置 ============

# 关注池ETF（当 Watchlist.md 解析失败时的兜底）
WATCHLIST_ETF = ["588000", "515980", "159995", "512690", "562500"]

# 数据模块开关（可按需启用）
MODULES = {
    "indices": True,      # 指数
    "holdings": True,     # 持仓行情
    "watchlist": True,    # 关注池行情
    "macro": True,        # 宏观数据
    "north_flow": True,   # 北向资金
    "sector": True,       # 行业/概念板块
    "fund_flow": True,    # 行业/概念资金流
    "news": True,         # 财联社快讯
    "technicals": True,   # 技术指标（MA/RSI/MACD/区间位置）
    "notices": False,     # 公告（较慢，默认关闭）
}

# 技术指标计算窗口
TECHNICAL_LOOKBACK_DAYS = 450  # 覆盖近一年交易日

# 行业/资金榜单数量
SECTOR_TOP_N = 20
FUND_FLOW_TOP_N = 20


# ============ 工具函数 ============

def log(msg):
    """输出日志到 stderr，不干扰 JSON 输出"""
    print(msg, file=sys.stderr)


def calc_pnl(cost, price):
    """计算盈亏百分比"""
    return round((price - cost) / cost * 100, 2) if cost > 0 else 0


def calc_days(buy_date):
    """计算持有天数"""
    try:
        return (datetime.now() - datetime.strptime(buy_date, "%Y-%m-%d")).days
    except:
        return 0


def safe_float(val, default=0):
    """安全转换为浮点数"""
    try:
        if pd.isna(val):
            return default
        return float(val)
    except:
        return default


def safe_round(val, digits=2):
    """安全四舍五入"""
    try:
        if val is None or pd.isna(val):
            return None
        return round(float(val), digits)
    except Exception:
        return None


def safe_int(val):
    """安全转换为整数"""
    try:
        if val is None or pd.isna(val):
            return None
        return int(val)
    except Exception:
        return None


def normalize_code(code, length):
    """标准化证券代码，保留前导 0"""
    if code is None:
        return ""
    return str(code).zfill(length)


def infer_asset_type(code, name, market):
    """根据代码/名称/市场推断标的类型"""
    code_str = str(code)
    name_str = name or ""
    market_str = market or ""

    if "ETF" in name_str or code_str.startswith(("5", "1")):
        return "ETF"
    if "港" in market_str or len(code_str) == 5:
        return "港股"
    return "A股"


def extract_table_rows(content, section_title):
    """提取 Markdown 表格行"""
    pattern = rf"## {re.escape(section_title)}\s*\n\s*\|[^\n]+\n\s*\|[-|\s]+\n((?:\|[^\n]+\n)*)"
    match = re.search(pattern, content)
    if not match:
        return []
    rows = match.group(1).strip().split("\n")
    parsed = []
    for row in rows:
        if not row.strip().startswith("|"):
            continue
        cols = [c.strip() for c in row.split("|")[1:-1]]
        parsed.append(cols)
    return parsed


def extract_list_items(content, section_title):
    """提取 Markdown 列表项"""
    header = f"## {section_title}"
    start = content.find(header)
    if start == -1:
        return []
    section = content[start + len(header):]
    next_header = re.search(r"^##\s+", section, flags=re.MULTILINE)
    if next_header:
        section = section[:next_header.start()]
    items = []
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
    return items


def latest_indicator_value(df, date_col="日期", value_col="今值"):
    """从宏观数据表中提取最新有效值"""
    if df is None or df.empty:
        return None
    try:
        temp = df[[date_col, value_col]].copy()
    except Exception:
        return None
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if temp.empty:
        return None
    latest = temp.iloc[-1]
    return {
        "date": latest[date_col].strftime("%Y-%m-%d"),
        "value": float(latest[value_col])
    }


def latest_date_str(*dates):
    """选择最新日期字符串"""
    valid = [pd.to_datetime(d, errors="coerce") for d in dates if d]
    valid = [d for d in valid if pd.notna(d)]
    if not valid:
        return ""
    return max(valid).strftime("%Y-%m-%d")


def pct_change(current, base):
    """计算百分比变化"""
    try:
        if base in (0, None) or pd.isna(base):
            return None
        return (float(current) / float(base) - 1) * 100
    except Exception:
        return None


def fetch_price_history(code, asset_type, start_date, end_date):
    """获取历史收盘价序列用于技术指标"""
    try:
        if asset_type == "ETF":
            df = ak.fund_etf_hist_em(
                symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust=""
            )
            date_col, close_col = "日期", "收盘"
        elif asset_type == "A股":
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust=""
            )
            date_col, close_col = "日期", "收盘"
        elif asset_type == "港股":
            df = ak.stock_hk_hist(
                symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust=""
            )
            date_col, close_col = "日期", "收盘"
        elif asset_type == "基金":
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            date_col, close_col = "净值日期", "单位净值"
        else:
            return None

        if df is None or df.empty:
            return None

        df = df[[date_col, close_col]].copy()
        df.rename(columns={date_col: "date", close_col: "close"}, inplace=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")

        if asset_type == "基金":
            start_dt = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
            if pd.notna(start_dt):
                df = df[df["date"] >= start_dt]

        return df
    except Exception as e:
        log(f"获取历史数据失败 {code} {asset_type}: {e}")
        return None


def compute_technicals(df):
    """计算常用技术指标"""
    if df is None or df.empty:
        return {"status": "missing"}

    close = df["close"]
    if close.empty:
        return {"status": "missing"}

    last = close.iloc[-1]
    data_points = int(close.shape[0])
    as_of = df["date"].iloc[-1]

    ma20 = close.rolling(20).mean().iloc[-1] if data_points >= 20 else None
    ma60 = close.rolling(60).mean().iloc[-1] if data_points >= 60 else None

    rsi14 = None
    if data_points >= 15:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi14 = rsi_series.iloc[-1]

    macd = macd_signal = macd_hist = None
    if data_points >= 26:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd = macd_line.iloc[-1]
        macd_signal = signal_line.iloc[-1]
        macd_hist = (macd_line - signal_line).iloc[-1]

    vol20 = None
    if data_points >= 21:
        returns = close.pct_change()
        vol20 = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100

    momentum_20d = None
    if data_points >= 21:
        momentum_20d = pct_change(last, close.iloc[-21])

    window = close.tail(252) if data_points >= 252 else close
    high_52w = window.max() if not window.empty else None
    low_52w = window.min() if not window.empty else None
    price_percentile_1y = None
    if high_52w is not None and low_52w is not None and high_52w != low_52w:
        price_percentile_1y = (last - low_52w) / (high_52w - low_52w) * 100

    trend = None
    if ma20 is not None and ma60 is not None:
        if last > ma20 and ma20 > ma60:
            trend = "上升"
        elif last < ma20 and ma20 < ma60:
            trend = "下降"
        else:
            trend = "震荡"

    return {
        "status": "ok",
        "as_of": as_of.strftime("%Y-%m-%d") if pd.notna(as_of) else "",
        "data_points": data_points,
        "ma20": safe_round(ma20, 4),
        "ma60": safe_round(ma60, 4),
        "price_vs_ma20_pct": safe_round(pct_change(last, ma20), 2),
        "price_vs_ma60_pct": safe_round(pct_change(last, ma60), 2),
        "rsi14": safe_round(rsi14, 2),
        "macd": safe_round(macd, 4),
        "macd_signal": safe_round(macd_signal, 4),
        "macd_hist": safe_round(macd_hist, 4),
        "volatility_20d": safe_round(vol20, 2),
        "momentum_20d": safe_round(momentum_20d, 2),
        "high_52w": safe_round(high_52w, 4),
        "low_52w": safe_round(low_52w, 4),
        "from_high_52w_pct": safe_round(pct_change(last, high_52w), 2),
        "from_low_52w_pct": safe_round(pct_change(last, low_52w), 2),
        "price_percentile_1y": safe_round(price_percentile_1y, 2),
        "trend": trend,
    }


def enrich_with_technicals(items):
    """为持仓/关注池补充技术指标"""
    if not items:
        return

    start_date = (datetime.now() - timedelta(days=TECHNICAL_LOOKBACK_DAYS)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    cache = {}

    for item in items:
        code = item.get("code")
        asset_type = item.get("type") or "ETF"
        if not code:
            continue
        key = (code, asset_type)
        if key not in cache:
            df = fetch_price_history(code, asset_type, start_date, end_date)
            cache[key] = compute_technicals(df)
        item["technicals"] = cache[key]


# ============ 持仓解析 ============

def parse_holdings_md():
    """从 Holdings.md 解析持仓配置"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    holdings_path = os.path.join(script_dir, "../..", "Config", "Holdings.md")

    holdings_etf = {}
    holdings_stock = {}
    holdings_hk = {}
    holdings_fund = {}

    if not os.path.exists(holdings_path):
        log(f"警告: Holdings.md 不存在: {holdings_path}")
        return holdings_etf, holdings_stock, holdings_hk, holdings_fund

    with open(holdings_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析A股持仓
    a_stock_match = re.search(r"## A股持仓\s*\n\s*\|[^\n]+\n\s*\|[-|\s]+\n((?:\|[^\n]+\n)*)", content)
    if a_stock_match:
        rows = a_stock_match.group(1).strip().split("\n")
        for row in rows:
            cols = [c.strip() for c in row.split("|")[1:-1]]
            if len(cols) >= 5:
                code, name, market, cost_str, qty_str, *rest = cols + [""] * 5
                try:
                    cost = float(cost_str)
                    qty = int(qty_str) if qty_str and qty_str != "-" else 0
                    buy_date = rest[1] if len(rest) > 1 and rest[1] and rest[1] != "-" else "2023-01-01"

                    if "ETF" in name or code.startswith("5") or code.startswith("1"):
                        holdings_etf[code] = {"name": name, "cost": cost, "qty": qty, "buy_date": buy_date}
                    else:
                        holdings_stock[code] = {"name": name, "cost": cost, "qty": qty, "buy_date": buy_date}
                except (ValueError, IndexError):
                    continue

    # 解析港股持仓
    hk_match = re.search(r"## 港股持仓\s*\n\s*\|[^\n]+\n\s*\|[-|\s]+\n((?:\|[^\n]+\n)*)", content)
    if hk_match:
        rows = hk_match.group(1).strip().split("\n")
        for row in rows:
            cols = [c.strip() for c in row.split("|")[1:-1]]
            if len(cols) >= 5:
                code, name, market, cost_str, qty_str, *rest = cols + [""] * 5
                try:
                    cost = float(cost_str)
                    qty = int(qty_str) if qty_str and qty_str != "-" else 0
                    buy_date = rest[1] if len(rest) > 1 and rest[1] and rest[1] != "-" else "2023-01-01"
                    holdings_hk[code] = {"name": name, "cost": cost, "qty": qty, "buy_date": buy_date}
                except (ValueError, IndexError):
                    continue

    # 解析基金持仓
    fund_match = re.search(r"## 基金持仓\s*\n\s*\|[^\n]+\n\s*\|[-|\s]+\n((?:\|[^\n]+\n)*)", content)
    if fund_match:
        rows = fund_match.group(1).strip().split("\n")
        for row in rows:
            cols = [c.strip() for c in row.split("|")[1:-1]]
            if len(cols) >= 6:
                code = cols[0]
                name = cols[1]
                cost_str = cols[4]
                qty_str = cols[5]
                buy_date = cols[7] if len(cols) > 7 and cols[7] and cols[7] != "-" else "2023-01-01"
                try:
                    cost = float(cost_str)
                    qty = float(qty_str) if qty_str and qty_str != "-" else 0
                    holdings_fund[code] = {"name": name, "cost": cost, "qty": qty, "buy_date": buy_date}
                except (ValueError, IndexError):
                    continue

    return holdings_etf, holdings_stock, holdings_hk, holdings_fund


# ============ 关注池解析 ============

def parse_watchlist_md():
    """从 Watchlist.md 解析关注池"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    watchlist_path = os.path.join(script_dir, "../..", "Config", "Watchlist.md")

    watchlist_items = []
    focus_industries = []
    excluded = []
    meta = {"source": "Config/Watchlist.md", "status": "ok"}

    if not os.path.exists(watchlist_path):
        log(f"警告: Watchlist.md 不存在: {watchlist_path}")
        meta["status"] = "missing"
        return watchlist_items, focus_industries, excluded, meta

    with open(watchlist_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 重点关注行业
    for cols in extract_table_rows(content, "重点关注行业"):
        if len(cols) < 5:
            continue
        focus_industries.append({
            "industry": cols[0],
            "reason": cols[1],
            "representative": cols[2],
            "logic": cols[3],
            "attitude": cols[4],
        })

    # 关注的个股/ETF
    for cols in extract_table_rows(content, "关注的个股/ETF"):
        if len(cols) < 7:
            continue
        code, name, market, reason, core_metrics, ideal_buy, status = cols[:7]
        watchlist_items.append({
            "code": code,
            "name": name,
            "market": market,
            "reason": reason,
            "core_metrics": core_metrics,
            "ideal_buy": ideal_buy,
            "status": status,
            "type": infer_asset_type(code, name, market),
        })

    # 排除清单
    excluded = extract_list_items(content, "排除清单")

    meta["watchlist_count"] = len(watchlist_items)
    meta["focus_count"] = len(focus_industries)
    meta["excluded_count"] = len(excluded)

    return watchlist_items, focus_industries, excluded, meta


def build_default_watchlist_items():
    """构建默认关注池（兜底）"""
    items = []
    for code in WATCHLIST_ETF:
        items.append({
            "code": code,
            "name": "",
            "market": "A股",
            "reason": "",
            "core_metrics": "",
            "ideal_buy": "",
            "status": "",
            "type": "ETF",
        })
    return items


def build_watchlist_base(item):
    """构建关注池基础字段"""
    return {
        "code": item.get("code", ""),
        "name": item.get("name", ""),
        "market": item.get("market", ""),
        "type": item.get("type", ""),
        "watch_reason": item.get("reason", ""),
        "core_metrics": item.get("core_metrics", ""),
        "ideal_buy": item.get("ideal_buy", ""),
        "status": item.get("status", ""),
    }


# ============ 数据获取模块 ============

def fetch_indices():
    """获取主要指数数据"""
    log("获取指数...")
    result = {}
    try:
        df = ak.stock_zh_index_spot_em()
        target_indices = ["上证指数", "深证成指", "沪深300", "中证500", "创业板指", "科创50"]
        for _, row in df.iterrows():
            if row['名称'] in target_indices:
                result[row['名称']] = {
                    "price": safe_float(row['最新价']),
                    "change": safe_float(row['涨跌幅'])
                }
    except Exception as e:
        log(f"获取指数失败: {e}")
    return result


def fetch_etf_data(holdings_etf, watchlist_items=None):
    """获取ETF数据"""
    log("获取ETF...")
    holdings_result = []
    watchlist_result = []
    watchlist_items = watchlist_items or []

    try:
        df = ak.fund_etf_spot_em()
        etf_data = {normalize_code(row['代码'], 6): row for _, row in df.iterrows()}

        # 持仓ETF
        for code, info in holdings_etf.items():
            code_key = normalize_code(code, 6)
            if code_key in etf_data:
                row = etf_data[code_key]
                price = safe_float(row['最新价'])
                holdings_result.append({
                    "code": code_key,
                    "name": info["name"],
                    "type": "ETF",
                    "price": price,
                    "change": safe_float(row['涨跌幅']),
                    "volume": safe_float(row.get('成交量', 0)),
                    "amount": safe_float(row.get('成交额', 0)),
                    "turnover_rate": safe_float(row.get('换手率', 0)),
                    "main_net_inflow": safe_float(row.get('主力净流入-净额', 0)),
                    "main_net_inflow_pct": safe_float(row.get('主力净流入-净占比', 0)),
                    "cost": info["cost"],
                    "qty": info.get("qty", 0),
                    "pnl": calc_pnl(info["cost"], price),
                    "days": calc_days(info["buy_date"])
                })

        # 关注池ETF
        for item in watchlist_items:
            code_key = normalize_code(item.get("code"), 6)
            if code_key in etf_data:
                row = etf_data[code_key]
                base = build_watchlist_base(item)
                base["code"] = code_key
                if not base["name"]:
                    base["name"] = str(row.get("名称", ""))
                base.update({
                    "price": safe_float(row.get('最新价'), default=None),
                    "change": safe_float(row.get('涨跌幅'), default=None),
                    "volume": safe_float(row.get('成交量'), default=None),
                    "amount": safe_float(row.get('成交额'), default=None),
                    "turnover_rate": safe_float(row.get('换手率'), default=None),
                    "main_net_inflow": safe_float(row.get('主力净流入-净额'), default=None),
                    "main_net_inflow_pct": safe_float(row.get('主力净流入-净占比'), default=None),
                })
                watchlist_result.append(base)
    except Exception as e:
        log(f"获取ETF失败: {e}")

    return holdings_result, watchlist_result


def fetch_a_stock_data(holdings_stock, watchlist_items=None):
    """获取A股个股数据"""
    log("获取A股...")
    holdings_result = []
    watchlist_result = []
    watchlist_items = watchlist_items or []
    try:
        df = ak.stock_zh_a_spot_em()
        stock_data = {normalize_code(row['代码'], 6): row for _, row in df.iterrows()}

        for code, info in holdings_stock.items():
            code_key = normalize_code(code, 6)
            if code_key in stock_data:
                row = stock_data[code_key]
                price = safe_float(row['最新价'])
                holdings_result.append({
                    "code": code_key,
                    "name": info["name"],
                    "type": "A股",
                    "price": price,
                    "change": safe_float(row['涨跌幅']),
                    "volume": safe_float(row.get('成交量', 0)),
                    "amount": safe_float(row.get('成交额', 0)),
                    "turnover_rate": safe_float(row.get('换手率', 0)),
                    "pe_ttm": safe_float(row.get('市盈率-动态', 0)),
                    "pb": safe_float(row.get('市净率', 0)),
                    "market_cap": safe_float(row.get('总市值', 0)),
                    "float_market_cap": safe_float(row.get('流通市值', 0)),
                    "cost": info["cost"],
                    "qty": info.get("qty", 0),
                    "pnl": calc_pnl(info["cost"], price),
                    "days": calc_days(info["buy_date"])
                })

        for item in watchlist_items:
            code_key = normalize_code(item.get("code"), 6)
            if code_key in stock_data:
                row = stock_data[code_key]
                base = build_watchlist_base(item)
                base["code"] = code_key
                if not base["name"]:
                    base["name"] = str(row.get("名称", ""))
                base.update({
                    "price": safe_float(row.get('最新价'), default=None),
                    "change": safe_float(row.get('涨跌幅'), default=None),
                    "volume": safe_float(row.get('成交量'), default=None),
                    "amount": safe_float(row.get('成交额'), default=None),
                    "turnover_rate": safe_float(row.get('换手率'), default=None),
                    "pe_ttm": safe_float(row.get('市盈率-动态'), default=None),
                    "pb": safe_float(row.get('市净率'), default=None),
                    "market_cap": safe_float(row.get('总市值'), default=None),
                    "float_market_cap": safe_float(row.get('流通市值'), default=None),
                    "main_net_inflow": safe_float(row.get('主力净流入-净额'), default=None),
                    "main_net_inflow_pct": safe_float(row.get('主力净流入-净占比'), default=None),
                })
                watchlist_result.append(base)
    except Exception as e:
        log(f"获取A股失败: {e}")
    return holdings_result, watchlist_result


def fetch_hk_stock_data(holdings_hk, watchlist_items=None):
    """获取港股数据"""
    log("获取港股...")
    holdings_result = []
    watchlist_result = []
    watchlist_items = watchlist_items or []
    try:
        df = ak.stock_hk_spot_em()
        hk_data = {normalize_code(row['代码'], 5): row for _, row in df.iterrows()}

        for code, info in holdings_hk.items():
            code_key = normalize_code(code, 5)
            if code_key in hk_data:
                row = hk_data[code_key]
                price = safe_float(row['最新价'])
                holdings_result.append({
                    "code": code_key,
                    "name": info["name"],
                    "type": "港股",
                    "price": price,
                    "change": safe_float(row['涨跌幅']),
                    "volume": safe_float(row.get('成交量', 0)),
                    "amount": safe_float(row.get('成交额', 0)),
                    "cost": info["cost"],
                    "qty": info.get("qty", 0),
                    "pnl": calc_pnl(info["cost"], price),
                    "days": calc_days(info["buy_date"])
                })

        for item in watchlist_items:
            code_key = normalize_code(item.get("code"), 5)
            if code_key in hk_data:
                row = hk_data[code_key]
                base = build_watchlist_base(item)
                base["code"] = code_key
                if not base["name"]:
                    base["name"] = str(row.get("名称", ""))
                base.update({
                    "price": safe_float(row.get('最新价'), default=None),
                    "change": safe_float(row.get('涨跌幅'), default=None),
                    "volume": safe_float(row.get('成交量'), default=None),
                    "amount": safe_float(row.get('成交额'), default=None),
                    "turnover_rate": safe_float(row.get('换手率'), default=None),
                })
                watchlist_result.append(base)
    except Exception as e:
        log(f"获取港股失败: {e}")
    return holdings_result, watchlist_result


def fetch_fund_data(holdings_fund):
    """获取基金净值数据"""
    log("获取基金...")
    result = []
    for code, info in holdings_fund.items():
        try:
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            if df is None or df.empty:
                continue
            latest = df.iloc[-1]
            price = safe_float(latest.get('单位净值', latest.iloc[1] if len(latest) > 1 else 0))
            change = safe_float(latest.get('日增长率', latest.iloc[2] if len(latest) > 2 else 0))
            result.append({
                "code": code,
                "name": info["name"],
                "type": "基金",
                "price": price,
                "change": change,
                "cost": info["cost"],
                "qty": info.get("qty", 0),
                "pnl": calc_pnl(info["cost"], price),
                "days": calc_days(info["buy_date"])
            })
        except Exception as e:
            log(f"获取基金失败 {code}: {e}")
    return result


def fetch_macro_data():
    """获取宏观经济数据"""
    log("获取宏观数据...")
    result = {}

    try:
        # 官方 PMI（制造业 + 非制造业）
        official_man = latest_indicator_value(ak.macro_china_pmi_yearly())
        official_non = latest_indicator_value(ak.macro_china_non_man_pmi())

        # 财新 PMI（制造业 + 服务业）
        caixin_man = latest_indicator_value(ak.macro_china_cx_pmi_yearly())
        caixin_srv = latest_indicator_value(ak.macro_china_cx_services_pmi_yearly())

        result["pmi"] = {
            "date": latest_date_str(
                official_man["date"] if official_man else "",
                official_non["date"] if official_non else ""
            ),
            "manufacturing": safe_float(official_man["value"], default=None) if official_man else None,
            "non_manufacturing": safe_float(official_non["value"], default=None) if official_non else None,
            "caixin_manufacturing": safe_float(caixin_man["value"], default=None) if caixin_man else None,
            "caixin_services": safe_float(caixin_srv["value"], default=None) if caixin_srv else None,
            "source": "AKShare"
        }
    except Exception as e:
        log(f"获取PMI失败: {e}")

    try:
        # CPI（同比 + 环比）
        cpi_yoy = latest_indicator_value(ak.macro_china_cpi_yearly())
        cpi_mom = latest_indicator_value(ak.macro_china_cpi_monthly())
        result["cpi"] = {
            "date": latest_date_str(
                cpi_yoy["date"] if cpi_yoy else "",
                cpi_mom["date"] if cpi_mom else ""
            ),
            "yoy": safe_float(cpi_yoy["value"], default=None) if cpi_yoy else None,
            "mom": safe_float(cpi_mom["value"], default=None) if cpi_mom else None,
            "source": "AKShare"
        }
    except Exception as e:
        log(f"获取CPI失败: {e}")

    try:
        # M2（同比）
        m2 = latest_indicator_value(ak.macro_china_m2_yearly())
        result["m2"] = {
            "date": m2["date"] if m2 else "",
            "yoy": safe_float(m2["value"], default=None) if m2 else None,
            "source": "AKShare"
        }
    except Exception as e:
        log(f"获取M2失败: {e}")

    return result


def fetch_north_flow():
    """获取北向资金数据"""
    log("获取北向资金...")
    result = {}

    try:
        # 当日汇总（优先使用汇总数据）
        df_sum = ak.stock_hsgt_fund_flow_summary_em()
        if df_sum is not None and not df_sum.empty and "资金方向" in df_sum.columns:
            north = df_sum[df_sum["资金方向"] == "北向"]
            if not north.empty:
                net_col = "成交净买额" if "成交净买额" in north.columns else "资金净流入"
                values = pd.to_numeric(north[net_col], errors="coerce")
                net_value = float(values.sum()) if values.notna().any() else None
                date = str(north["交易日"].iloc[0]) if "交易日" in north.columns else ""
                result["today"] = {
                    "date": date,
                    "net_flow": net_value,
                    "source": "summary"
                }
    except Exception as e:
        log(f"获取北向资金失败(汇总): {e}")

    try:
        # 历史 5 日（可能存在滞后）
        df = ak.stock_hsgt_hist_em()
        if df is not None and not df.empty:
            if "日期" in df.columns:
                df = df.sort_values("日期")
            net_col = "当日成交净买额" if "当日成交净买额" in df.columns else "当日资金流入"
            valid = df[df[net_col].notna()] if net_col in df.columns else df
            result["recent_5_days"] = []
            if not valid.empty:
                recent = valid.tail(5)
                for _, row in recent.iterrows():
                    result["recent_5_days"].append({
                        "date": str(row.get("日期", "")),
                        "net_flow": safe_float(row.get(net_col), default=None)
                    })
                last_date = str(recent.iloc[-1].get("日期", ""))
                result["history_last_date"] = last_date
                try:
                    last_dt = pd.to_datetime(last_date, errors="coerce")
                    if pd.notna(last_dt):
                        stale_days = (datetime.now().date() - last_dt.date()).days
                        result["history_stale_days"] = stale_days
                        if stale_days > 7:
                            result["history_status"] = "stale"
                except Exception:
                    pass
    except Exception as e:
        log(f"获取北向资金失败: {e}")

    try:
        # 分时数据作为补充
        df_min = ak.stock_hsgt_fund_min_em(symbol="北向资金")
        if df_min is not None and not df_min.empty and "北向资金" in df_min.columns:
            df_min = df_min[df_min["北向资金"].notna()]
            if not df_min.empty:
                latest = df_min.iloc[-1]
                result["intraday"] = {
                    "date": str(latest.get("日期", "")),
                    "time": str(latest.get("时间", "")),
                    "net_flow": safe_float(latest.get("北向资金"), default=None),
                    "source": "min"
                }
    except Exception as e:
        log(f"获取北向资金失败(分时): {e}")

    return result


def fetch_sector_rank(top_n=SECTOR_TOP_N):
    """获取行业/概念板块排名"""
    log("获取行业/概念板块...")
    result = {"industry": [], "concept": [], "source": "AKShare"}

    try:
        df = ak.stock_board_industry_name_em()
        if df is not None and not df.empty:
            for _, row in df.head(top_n).iterrows():
                result["industry"].append({
                    "rank": safe_int(row.get("排名")),
                    "name": str(row.get("板块名称", "")),
                    "code": str(row.get("板块代码", "")),
                    "price": safe_float(row.get("最新价"), default=None),
                    "change": safe_float(row.get("涨跌幅"), default=None),
                    "turnover_rate": safe_float(row.get("换手率"), default=None),
                    "up": safe_int(row.get("上涨家数")),
                    "down": safe_int(row.get("下跌家数")),
                    "leader": str(row.get("领涨股票", "")),
                    "leader_change": safe_float(row.get("领涨股票-涨跌幅"), default=None),
                })
    except Exception as e:
        log(f"获取行业板块失败: {e}")

    try:
        df = ak.stock_board_concept_name_em()
        if df is not None and not df.empty:
            for _, row in df.head(top_n).iterrows():
                result["concept"].append({
                    "rank": safe_int(row.get("排名")),
                    "name": str(row.get("板块名称", "")),
                    "code": str(row.get("板块代码", "")),
                    "price": safe_float(row.get("最新价"), default=None),
                    "change": safe_float(row.get("涨跌幅"), default=None),
                    "turnover_rate": safe_float(row.get("换手率"), default=None),
                    "up": safe_int(row.get("上涨家数")),
                    "down": safe_int(row.get("下跌家数")),
                    "leader": str(row.get("领涨股票", "")),
                    "leader_change": safe_float(row.get("领涨股票-涨跌幅"), default=None),
                })
    except Exception as e:
        log(f"获取概念板块失败: {e}")

    return result


def fetch_sector_fund_flow(top_n=FUND_FLOW_TOP_N):
    """获取行业/概念资金流向"""
    log("获取行业/概念资金流...")
    result = {"industry": [], "concept": [], "source": "AKShare"}

    try:
        df = ak.stock_fund_flow_industry()
        if df is not None and not df.empty:
            for _, row in df.head(top_n).iterrows():
                result["industry"].append({
                    "rank": safe_int(row.get("序号")),
                    "name": str(row.get("行业", "")),
                    "index": safe_float(row.get("行业指数"), default=None),
                    "change": safe_float(row.get("行业-涨跌幅"), default=None),
                    "inflow": safe_float(row.get("流入资金"), default=None),
                    "outflow": safe_float(row.get("流出资金"), default=None),
                    "net_flow": safe_float(row.get("净额"), default=None),
                    "company_count": safe_int(row.get("公司家数")),
                    "leader": str(row.get("领涨股", "")),
                    "leader_change": safe_float(row.get("领涨股-涨跌幅"), default=None),
                    "price": safe_float(row.get("当前价"), default=None),
                })
    except Exception as e:
        log(f"获取行业资金流失败: {e}")

    try:
        df = ak.stock_fund_flow_concept()
        if df is not None and not df.empty:
            for _, row in df.head(top_n).iterrows():
                result["concept"].append({
                    "rank": safe_int(row.get("序号")),
                    "name": str(row.get("行业", "")),
                    "index": safe_float(row.get("行业指数"), default=None),
                    "change": safe_float(row.get("行业-涨跌幅"), default=None),
                    "inflow": safe_float(row.get("流入资金"), default=None),
                    "outflow": safe_float(row.get("流出资金"), default=None),
                    "net_flow": safe_float(row.get("净额"), default=None),
                    "company_count": safe_int(row.get("公司家数")),
                    "leader": str(row.get("领涨股", "")),
                    "leader_change": safe_float(row.get("领涨股-涨跌幅"), default=None),
                    "price": safe_float(row.get("当前价"), default=None),
                })
    except Exception as e:
        log(f"获取概念资金流失败: {e}")

    return result


def fetch_cls_news(limit=20):
    """获取财联社电报快讯（替代WebSearch）"""
    log("获取财联社快讯...")
    result = []

    try:
        df = ak.stock_info_global_cls()
        if not df.empty:
            for _, row in df.head(limit).iterrows():
                result.append({
                    "time": str(row.get('发布时间', row.get('时间', ''))),
                    "title": str(row.get('标题', row.get('内容', '')))[:200],
                    "importance": str(row.get('重要性', '普通'))
                })
    except Exception as e:
        log(f"获取财联社快讯失败: {e}")

    # 备用：东方财富快讯
    if not result:
        try:
            df = ak.stock_info_global_em()
            if not df.empty:
                for _, row in df.head(limit).iterrows():
                    result.append({
                        "time": str(row.get('时间', '')),
                        "title": str(row.get('内容', ''))[:200],
                        "importance": "普通"
                    })
        except Exception as e:
            log(f"获取东财快讯失败: {e}")

    return result


def fetch_stock_notices(stock_codes):
    """获取个股公告（可选，较慢）"""
    log("获取公告...")
    result = []

    for code in stock_codes[:5]:  # 限制数量
        try:
            df = ak.stock_notice_report(symbol=code)
            if not df.empty:
                for _, row in df.head(3).iterrows():
                    result.append({
                        "code": code,
                        "date": str(row.get('公告日期', '')),
                        "title": str(row.get('公告标题', ''))[:100]
                    })
        except:
            continue

    return result


# ============ 主函数 ============

def main():
    # 从 Holdings.md 读取持仓配置
    holdings_etf, holdings_stock, holdings_hk, holdings_fund = parse_holdings_md()

    log(f"解析到持仓: ETF={len(holdings_etf)}, A股={len(holdings_stock)}, 港股={len(holdings_hk)}, 基金={len(holdings_fund)}")

    # 从 Watchlist.md 读取关注池配置
    watchlist_items, focus_industries, excluded, watchlist_meta = parse_watchlist_md()
    if not watchlist_items:
        watchlist_items = build_default_watchlist_items()
        watchlist_meta["fallback"] = "default_watchlist_etf"

    result = {
        "metadata": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "version": "2.0",
            "source": "AKShare",
            "technicals": {
                "lookback_days": TECHNICAL_LOOKBACK_DAYS,
                "note": "基于日线收盘价计算"
            }
        }
    }

    if MODULES["watchlist"]:
        result["watchlist_meta"] = {
            **watchlist_meta,
            "focus_industries": focus_industries,
            "excluded": excluded
        }

    # 1. 指数数据
    if MODULES["indices"]:
        result["indices"] = fetch_indices()

    # 2. 持仓数据
    if MODULES["holdings"]:
        result["holdings"] = []
        watchlist_etf = [item for item in watchlist_items if item.get("type") == "ETF"]
        watchlist_a = [item for item in watchlist_items if item.get("type") == "A股"]
        watchlist_hk = [item for item in watchlist_items if item.get("type") == "港股"]

        # ETF
        etf_holdings, etf_watchlist = fetch_etf_data(
            holdings_etf,
            watchlist_etf if MODULES["watchlist"] else []
        )
        result["holdings"].extend(etf_holdings)

        # A股
        a_stock_holdings, a_stock_watchlist = fetch_a_stock_data(
            holdings_stock,
            watchlist_a if MODULES["watchlist"] else []
        )
        result["holdings"].extend(a_stock_holdings)

        # 港股
        hk_stock_holdings, hk_stock_watchlist = fetch_hk_stock_data(
            holdings_hk,
            watchlist_hk if MODULES["watchlist"] else []
        )
        result["holdings"].extend(hk_stock_holdings)

        # 基金
        fund_data = fetch_fund_data(holdings_fund)
        result["holdings"].extend(fund_data)

        if MODULES["watchlist"]:
            result["watchlist"] = []
            result["watchlist"].extend(etf_watchlist)
            result["watchlist"].extend(a_stock_watchlist)
            result["watchlist"].extend(hk_stock_watchlist)

        # 技术指标
        if MODULES["technicals"]:
            enrich_with_technicals(result["holdings"])
            if MODULES["watchlist"] and "watchlist" in result:
                enrich_with_technicals(result["watchlist"])

    # 3. 宏观数据
    if MODULES["macro"]:
        result["macro"] = fetch_macro_data()

    # 4. 北向资金
    if MODULES["north_flow"]:
        result["north_flow"] = fetch_north_flow()

    # 5. 行业/概念板块
    if MODULES["sector"]:
        result["sector"] = fetch_sector_rank()

    # 6. 行业/概念资金流
    if MODULES["fund_flow"]:
        result["fund_flow"] = fetch_sector_fund_flow()

    # 7. 快讯
    if MODULES["news"]:
        result["news"] = fetch_cls_news(15)

    # 8. 公告（可选）
    if MODULES["notices"]:
        all_codes = list(holdings_stock.keys()) + [code for code in holdings_hk.keys()]
        result["notices"] = fetch_stock_notices(all_codes)

    # 输出 JSON
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
