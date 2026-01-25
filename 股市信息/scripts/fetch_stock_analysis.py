"""
个股深度分析数据获取脚本 v2.0
专门为 /analyze 技能提供数据支持

核心改进：
1. 成交量分析：量比、量价配合、量价背离
2. 多周期分析：日线+周线趋势共振
3. 改进的支撑压力位：均线+整数关口+前高前低
4. ATR动态止损
5. 更科学的评分体系

用法：
    python3 fetch_stock_analysis.py 588000           # 分析ETF
    python3 fetch_stock_analysis.py 002594           # 分析A股
    python3 fetch_stock_analysis.py 00700 --market hk  # 分析港股
"""

import akshare as ak
import pandas as pd
import numpy as np
import json
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


def log(msg):
    """输出日志到 stderr"""
    print(msg, file=sys.stderr)


def safe_float(val, default=0.0):
    """安全转换为浮点数"""
    try:
        if pd.isna(val):
            return default
        return round(float(val), 4)
    except:
        return default


# ============ 技术指标计算 ============

def calc_ma(prices, period):
    """计算移动平均线"""
    return prices.rolling(window=period).mean()


def calc_ema(prices, period):
    """计算指数移动平均"""
    return prices.ewm(span=period, adjust=False).mean()


def calc_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD"""
    ema_fast = calc_ema(prices, fast)
    ema_slow = calc_ema(prices, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd = (dif - dea) * 2
    return dif, dea, macd


def calc_rsi(prices, period=14):
    """计算RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_bollinger(prices, period=20, std_dev=2):
    """计算布林带"""
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, ma, lower


def calc_atr(df, period=14):
    """计算ATR（真实波动幅度）"""
    high = df['high']
    low = df['low']
    close = df['close']

    # 真实波动幅度
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    return atr


# ============ 成交量分析（新增） ============

def calc_volume_analysis(df):
    """成交量分析"""
    if 'volume' not in df.columns or len(df) < 20:
        return {"error": "成交量数据不足"}

    volume = df['volume']
    close = df['close']

    # 今日成交量
    today_vol = volume.iloc[-1]

    # 均量
    vol_ma5 = volume.rolling(5).mean().iloc[-1]
    vol_ma20 = volume.rolling(20).mean().iloc[-1]

    # 量比 = 今日成交量 / 5日均量
    volume_ratio = today_vol / vol_ma5 if vol_ma5 > 0 else 0

    # 量价配合判断
    price_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0
    vol_change = (today_vol / volume.iloc[-2] - 1) * 100 if len(volume) >= 2 else 0

    # 量价关系判断
    if price_change > 0.5 and vol_change > 20:
        vol_price_relation = "放量上涨"
        vol_price_signal = "量价配合，健康"
    elif price_change > 0.5 and vol_change < -20:
        vol_price_relation = "缩量上涨"
        vol_price_signal = "上涨乏力，警惕"
    elif price_change < -0.5 and vol_change > 20:
        vol_price_relation = "放量下跌"
        vol_price_signal = "恐慌抛售，观望"
    elif price_change < -0.5 and vol_change < -20:
        vol_price_relation = "缩量下跌"
        vol_price_signal = "惜售，可能企稳"
    else:
        vol_price_relation = "量价平稳"
        vol_price_signal = "观望"

    # 量价背离检测（近5日）
    divergence = None
    if len(df) >= 5:
        recent_price_trend = close.iloc[-1] > close.iloc[-5]
        recent_vol_trend = volume.iloc[-1] > volume.iloc[-5]

        if recent_price_trend and not recent_vol_trend:
            divergence = "价涨量缩背离"
        elif not recent_price_trend and recent_vol_trend:
            divergence = "价跌量增背离"

    # 量能状态
    if volume_ratio > 2:
        vol_status = "显著放量"
    elif volume_ratio > 1.5:
        vol_status = "温和放量"
    elif volume_ratio > 0.8:
        vol_status = "量能正常"
    elif volume_ratio > 0.5:
        vol_status = "缩量"
    else:
        vol_status = "极度缩量"

    return {
        "today_volume": safe_float(today_vol),
        "vol_ma5": safe_float(vol_ma5),
        "vol_ma20": safe_float(vol_ma20),
        "volume_ratio": safe_float(volume_ratio),
        "vol_status": vol_status,
        "vol_price_relation": vol_price_relation,
        "vol_price_signal": vol_price_signal,
        "divergence": divergence
    }


# ============ 多周期分析（新增） ============

def resample_to_weekly(df):
    """将日线数据重采样为周线"""
    df = df.copy()
    df.set_index('date', inplace=True)

    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    weekly = weekly.reset_index()
    return weekly


def calc_multi_timeframe_trend(df_daily, df_weekly):
    """多周期趋势分析"""
    result = {
        "daily": {},
        "weekly": {},
        "resonance": None
    }

    # 日线趋势
    if len(df_daily) >= 60:
        close_d = df_daily['close']
        ma5_d = calc_ma(close_d, 5).iloc[-1]
        ma10_d = calc_ma(close_d, 10).iloc[-1]
        ma20_d = calc_ma(close_d, 20).iloc[-1]
        ma60_d = calc_ma(close_d, 60).iloc[-1]
        current_d = close_d.iloc[-1]

        if ma5_d > ma10_d > ma20_d > ma60_d:
            daily_trend = "多头排列"
            daily_score = 2
        elif ma5_d < ma10_d < ma20_d < ma60_d:
            daily_trend = "空头排列"
            daily_score = -2
        elif current_d > ma20_d:
            daily_trend = "偏多"
            daily_score = 1
        elif current_d < ma20_d:
            daily_trend = "偏空"
            daily_score = -1
        else:
            daily_trend = "震荡"
            daily_score = 0

        result["daily"] = {
            "trend": daily_trend,
            "score": daily_score,
            "price_vs_ma20": safe_float((current_d / ma20_d - 1) * 100)
        }

    # 周线趋势
    if len(df_weekly) >= 20:
        close_w = df_weekly['close']
        ma5_w = calc_ma(close_w, 5).iloc[-1]
        ma10_w = calc_ma(close_w, 10).iloc[-1]
        ma20_w = calc_ma(close_w, 20).iloc[-1]
        current_w = close_w.iloc[-1]

        if ma5_w > ma10_w > ma20_w:
            weekly_trend = "周线多头"
            weekly_score = 2
        elif ma5_w < ma10_w < ma20_w:
            weekly_trend = "周线空头"
            weekly_score = -2
        elif current_w > ma10_w:
            weekly_trend = "周线偏多"
            weekly_score = 1
        elif current_w < ma10_w:
            weekly_trend = "周线偏空"
            weekly_score = -1
        else:
            weekly_trend = "周线震荡"
            weekly_score = 0

        result["weekly"] = {
            "trend": weekly_trend,
            "score": weekly_score,
            "price_vs_ma10": safe_float((current_w / ma10_w - 1) * 100)
        }

    # 共振判断
    if result["daily"] and result["weekly"]:
        d_score = result["daily"].get("score", 0)
        w_score = result["weekly"].get("score", 0)

        if d_score > 0 and w_score > 0:
            result["resonance"] = "日周共振向上"
            result["resonance_signal"] = "强势，可积极参与"
        elif d_score < 0 and w_score < 0:
            result["resonance"] = "日周共振向下"
            result["resonance_signal"] = "弱势，回避"
        elif d_score > 0 and w_score < 0:
            result["resonance"] = "日线反弹周线空"
            result["resonance_signal"] = "反弹，谨慎参与"
        elif d_score < 0 and w_score > 0:
            result["resonance"] = "日线回调周线多"
            result["resonance_signal"] = "回调买点，可关注"
        else:
            result["resonance"] = "趋势不明"
            result["resonance_signal"] = "观望"

    return result


# ============ 改进的支撑压力位 ============

def calc_support_resistance_v2(df, ma_values):
    """改进的支撑压力位计算"""
    if len(df) < 60:
        return {"error": "数据不足"}

    current = df['close'].iloc[-1]
    recent = df.tail(60)

    levels = []

    # 1. 近期高低点
    high_max = recent['high'].max()
    low_min = recent['low'].min()

    # 2. 前高前低（取前10个极值点）
    highs = recent['high'].nlargest(5).values
    lows = recent['low'].nsmallest(5).values

    # 3. 均线位置
    ma20 = ma_values.get('ma20', 0)
    ma60 = ma_values.get('ma60', 0)

    # 4. 整数关口（基于当前价格）
    base = current
    integer_levels = []
    for pct in [-10, -5, 5, 10]:
        level = round(base * (1 + pct / 100), 2)
        # 取整到合理的整数
        if level > 100:
            level = round(level / 10) * 10
        elif level > 10:
            level = round(level)
        else:
            level = round(level, 1)
        integer_levels.append(level)

    # 汇总所有水平
    all_levels = list(highs) + list(lows) + [ma20, ma60] + integer_levels
    all_levels = [l for l in all_levels if l > 0]

    # 分类：压力位（高于现价）和支撑位（低于现价）
    resistance = sorted([l for l in all_levels if l > current * 1.01])[:3]
    support = sorted([l for l in all_levels if l < current * 0.99], reverse=True)[:3]

    # 计算与现价的距离
    result = {
        "current": safe_float(current),
        "resistance": [],
        "support": []
    }

    for r in resistance:
        result["resistance"].append({
            "price": safe_float(r),
            "distance_pct": safe_float((r / current - 1) * 100)
        })

    for s in support:
        result["support"].append({
            "price": safe_float(s),
            "distance_pct": safe_float((s / current - 1) * 100)
        })

    # 简化版（兼容旧格式）
    result["resistance_1"] = result["resistance"][0]["price"] if len(result["resistance"]) > 0 else safe_float(high_max)
    result["resistance_2"] = result["resistance"][1]["price"] if len(result["resistance"]) > 1 else safe_float(high_max * 1.05)
    result["support_1"] = result["support"][0]["price"] if len(result["support"]) > 0 else safe_float(ma20)
    result["support_2"] = result["support"][1]["price"] if len(result["support"]) > 1 else safe_float(low_min)

    return result


# ============ 区间表现 ============

def calc_performance(df):
    """计算区间涨跌幅"""
    close = df['close']
    current = close.iloc[-1]

    result = {}

    # 今日涨跌
    if len(close) >= 2:
        result["today"] = safe_float((current / close.iloc[-2] - 1) * 100)

    # 近5日
    if len(close) >= 5:
        result["5d"] = safe_float((current / close.iloc[-5] - 1) * 100)

    # 近20日（约1个月）
    if len(close) >= 20:
        result["20d"] = safe_float((current / close.iloc[-20] - 1) * 100)

    # 近60日（约3个月）
    if len(close) >= 60:
        result["60d"] = safe_float((current / close.iloc[-60] - 1) * 100)

    # 年初至今
    year_start = df[df['date'].dt.year == datetime.now().year]
    if len(year_start) > 0:
        result["ytd"] = safe_float((current / year_start['close'].iloc[0] - 1) * 100)

    return result


# ============ 数据获取模块 ============

def fetch_a_stock_kline(code, days=250):
    """获取A股K线数据"""
    log(f"获取A股日K线: {code}")
    try:
        # 判断是否是ETF
        if code.startswith('5') or code.startswith('1'):
            df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
        else:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")

        if df.empty:
            return None

        # 标准化列名
        df.columns = [c.lower() for c in df.columns]
        if '日期' in df.columns:
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                    '最高': 'high', '最低': 'low', '成交量': 'volume'})

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(days)

        return df
    except Exception as e:
        log(f"获取A股K线失败: {e}")
        return None


def fetch_hk_stock_kline(code, days=250):
    """获取港股K线数据"""
    log(f"获取港股日K线: {code}")
    try:
        df = ak.stock_hk_hist(symbol=code, period="daily", adjust="qfq")

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        if '日期' in df.columns:
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                    '最高': 'high', '最低': 'low', '成交量': 'volume'})

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(days)

        return df
    except Exception as e:
        log(f"获取港股K线失败: {e}")
        return None


def fetch_fund_flow(code):
    """获取资金流向"""
    log(f"获取资金流向: {code}")
    result = {}

    try:
        # 个股资金流向
        df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith('6') else "sz")

        if not df.empty:
            latest = df.iloc[-1]
            result["today"] = {
                "main_net": safe_float(latest.get('主力净流入-净额', 0)),
                "main_ratio": safe_float(latest.get('主力净流入-净占比', 0)),
                "retail_net": safe_float(latest.get('散户净流入-净额', 0)),
            }

            # 近5日汇总
            recent5 = df.tail(5)
            result["5d_total"] = {
                "main_net": safe_float(recent5.get('主力净流入-净额', pd.Series([0])).sum()),
            }

            # 连续流入/流出天数
            main_net_col = '主力净流入-净额'
            if main_net_col in df.columns:
                consecutive = 0
                direction = None
                for i in range(len(df) - 1, max(len(df) - 10, -1), -1):
                    val = df.iloc[i].get(main_net_col, 0)
                    if val > 0:
                        if direction == "in" or direction is None:
                            consecutive += 1
                            direction = "in"
                        else:
                            break
                    elif val < 0:
                        if direction == "out" or direction is None:
                            consecutive += 1
                            direction = "out"
                        else:
                            break
                    else:
                        break
                result["consecutive"] = {
                    "days": consecutive,
                    "direction": "流入" if direction == "in" else "流出" if direction == "out" else "无"
                }
    except Exception as e:
        log(f"获取资金流向失败: {e}")

    return result


def fetch_financial_data(code):
    """获取财务数据"""
    log(f"获取财务数据: {code}")
    result = {}

    try:
        # 财务摘要
        df = ak.stock_financial_abstract(symbol=code)

        if not df.empty:
            # 获取最近几期数据
            recent = df.head(4)  # 最近4期

            for idx, row in recent.iterrows():
                period = str(row.get('报告期', idx))
                result[period] = {
                    "revenue": safe_float(row.get('营业总收入', 0)),
                    "net_profit": safe_float(row.get('净利润', 0)),
                    "roe": safe_float(row.get('净资产收益率', 0)),
                    "gross_margin": safe_float(row.get('销售毛利率', 0)),
                    "debt_ratio": safe_float(row.get('资产负债率', 0)),
                }
    except Exception as e:
        log(f"获取财务数据失败: {e}")

    return result


def fetch_valuation(code):
    """获取估值数据"""
    log(f"获取估值数据: {code}")
    result = {}

    try:
        # 使用市盈率等指标接口
        df = ak.stock_a_lg_indicator(symbol=code)

        if not df.empty:
            latest = df.iloc[-1]
            result["current"] = {
                "pe_ttm": safe_float(latest.get('pe_ttm', latest.get('pe', 0))),
                "pb": safe_float(latest.get('pb', 0)),
                "ps_ttm": safe_float(latest.get('ps_ttm', latest.get('ps', 0))),
                "total_mv": safe_float(latest.get('total_mv', 0)),  # 总市值（亿）
            }

            # 计算历史分位（如果有足够数据）
            if len(df) >= 250:
                pe_series = df['pe_ttm'] if 'pe_ttm' in df.columns else df.get('pe', pd.Series())
                if not pe_series.empty:
                    current_pe = result["current"]["pe_ttm"]
                    if current_pe > 0:
                        percentile = (pe_series < current_pe).sum() / len(pe_series) * 100
                        result["pe_percentile"] = safe_float(percentile)
    except Exception as e:
        log(f"获取估值数据失败: {e}")

    return result


def fetch_etf_info(code):
    """获取ETF信息"""
    log(f"获取ETF信息: {code}")
    result = {}

    # 方法1: 尝试获取ETF持仓明细
    try:
        df = ak.fund_portfolio_hold_em(symbol=code, date="2024")
        if not df.empty:
            result["holdings"] = []
            for _, row in df.head(10).iterrows():
                result["holdings"].append({
                    "name": str(row.get('股票名称', row.get('证券名称', ''))),
                    "code": str(row.get('股票代码', row.get('证券代码', ''))),
                    "weight": safe_float(row.get('占净值比例', row.get('持仓市值占比', 0)))
                })
            return result
    except Exception as e:
        log(f"方法1获取ETF持仓失败: {e}")

    # 方法2: 使用场内ETF成份股接口
    try:
        df = ak.fund_etf_spot_em()
        etf_row = df[df['代码'] == code]
        if not etf_row.empty:
            result["fund_name"] = str(etf_row.iloc[0].get('名称', ''))
            result["nav"] = safe_float(etf_row.iloc[0].get('最新价', 0))
    except Exception as e:
        log(f"方法2获取ETF信息失败: {e}")

    return result


def fetch_dragon_tiger(code):
    """获取龙虎榜数据"""
    log(f"获取龙虎榜: {code}")
    result = []

    try:
        df = ak.stock_lhb_stock_statistic_em(symbol=code)
        if not df.empty:
            for _, row in df.head(5).iterrows():
                result.append({
                    "date": str(row.get('上榜日期', '')),
                    "reason": str(row.get('上榜原因', '')),
                    "buy_total": safe_float(row.get('买入总额', 0)),
                    "sell_total": safe_float(row.get('卖出总额', 0)),
                })
    except Exception as e:
        log(f"获取龙虎榜失败: {e}")

    return result


# ============ 综合评分系统（优化版） ============

def calc_comprehensive_score(data):
    """
    综合评分系统 v2.0

    评分维度：
    1. 趋势分（30分）：日线趋势 + 周线趋势 + 共振
    2. 动能分（25分）：MACD + RSI
    3. 量能分（20分）：量价配合
    4. 位置分（15分）：布林带位置 + 与均线距离
    5. 资金分（10分）：主力资金流向

    总分100分
    """
    scores = {
        "trend": {"score": 0, "max": 30, "detail": []},
        "momentum": {"score": 0, "max": 25, "detail": []},
        "volume": {"score": 0, "max": 20, "detail": []},
        "position": {"score": 0, "max": 15, "detail": []},
        "fund": {"score": 0, "max": 10, "detail": []},
    }

    # 1. 趋势分（30分）
    multi_tf = data.get("multi_timeframe", {})

    # 日线趋势（10分）
    daily_score = multi_tf.get("daily", {}).get("score", 0)
    if daily_score == 2:
        scores["trend"]["score"] += 10
        scores["trend"]["detail"].append("日线多头排列(+10)")
    elif daily_score == 1:
        scores["trend"]["score"] += 6
        scores["trend"]["detail"].append("日线偏多(+6)")
    elif daily_score == -1:
        scores["trend"]["score"] += 3
        scores["trend"]["detail"].append("日线偏空(+3)")
    elif daily_score == -2:
        scores["trend"]["score"] += 0
        scores["trend"]["detail"].append("日线空头排列(+0)")
    else:
        scores["trend"]["score"] += 5
        scores["trend"]["detail"].append("日线震荡(+5)")

    # 周线趋势（10分）
    weekly_score = multi_tf.get("weekly", {}).get("score", 0)
    if weekly_score == 2:
        scores["trend"]["score"] += 10
        scores["trend"]["detail"].append("周线多头(+10)")
    elif weekly_score == 1:
        scores["trend"]["score"] += 7
        scores["trend"]["detail"].append("周线偏多(+7)")
    elif weekly_score == -1:
        scores["trend"]["score"] += 3
        scores["trend"]["detail"].append("周线偏空(+3)")
    elif weekly_score == -2:
        scores["trend"]["score"] += 0
        scores["trend"]["detail"].append("周线空头(+0)")
    else:
        scores["trend"]["score"] += 5
        scores["trend"]["detail"].append("周线震荡(+5)")

    # 共振加分（10分）
    resonance = multi_tf.get("resonance", "")
    if "共振向上" in resonance:
        scores["trend"]["score"] += 10
        scores["trend"]["detail"].append("日周共振向上(+10)")
    elif "回调周线多" in resonance:
        scores["trend"]["score"] += 7
        scores["trend"]["detail"].append("回调买点(+7)")
    elif "反弹周线空" in resonance:
        scores["trend"]["score"] += 3
        scores["trend"]["detail"].append("反弹非底(+3)")
    elif "共振向下" in resonance:
        scores["trend"]["score"] += 0
        scores["trend"]["detail"].append("日周共振向下(+0)")
    else:
        scores["trend"]["score"] += 5
        scores["trend"]["detail"].append("趋势不明(+5)")

    # 2. 动能分（25分）
    macd = data.get("macd", {})
    rsi = data.get("rsi", {})

    # MACD（15分）
    macd_signal = macd.get("signal", "")
    if macd_signal == "金叉":
        scores["momentum"]["score"] += 15
        scores["momentum"]["detail"].append("MACD金叉(+15)")
    elif macd_signal == "多头":
        dif = macd.get("dif", 0)
        if dif > 0:
            scores["momentum"]["score"] += 12
            scores["momentum"]["detail"].append("MACD多头零轴上(+12)")
        else:
            scores["momentum"]["score"] += 8
            scores["momentum"]["detail"].append("MACD多头零轴下(+8)")
    elif macd_signal == "死叉":
        scores["momentum"]["score"] += 2
        scores["momentum"]["detail"].append("MACD死叉(+2)")
    else:  # 空头
        scores["momentum"]["score"] += 5
        scores["momentum"]["detail"].append("MACD空头(+5)")

    # RSI（10分）
    rsi_val = rsi.get("value", 50)
    if 40 <= rsi_val <= 60:
        scores["momentum"]["score"] += 7
        scores["momentum"]["detail"].append(f"RSI中性{rsi_val:.0f}(+7)")
    elif 30 <= rsi_val < 40:
        scores["momentum"]["score"] += 9
        scores["momentum"]["detail"].append(f"RSI偏低{rsi_val:.0f}(+9)")
    elif 20 <= rsi_val < 30:
        scores["momentum"]["score"] += 10
        scores["momentum"]["detail"].append(f"RSI超卖{rsi_val:.0f}(+10)")
    elif rsi_val < 20:
        scores["momentum"]["score"] += 8
        scores["momentum"]["detail"].append(f"RSI极度超卖{rsi_val:.0f}(+8)")
    elif 60 < rsi_val <= 70:
        scores["momentum"]["score"] += 6
        scores["momentum"]["detail"].append(f"RSI偏高{rsi_val:.0f}(+6)")
    elif 70 < rsi_val <= 80:
        scores["momentum"]["score"] += 4
        scores["momentum"]["detail"].append(f"RSI超买{rsi_val:.0f}(+4)")
    else:
        scores["momentum"]["score"] += 2
        scores["momentum"]["detail"].append(f"RSI极度超买{rsi_val:.0f}(+2)")

    # 3. 量能分（20分）
    vol = data.get("volume_analysis", {})
    if vol and "error" not in vol:
        vol_ratio = vol.get("volume_ratio", 1)
        vol_signal = vol.get("vol_price_signal", "")
        divergence = vol.get("divergence")

        # 量比（10分）
        if 1.0 <= vol_ratio <= 2.0:
            scores["volume"]["score"] += 10
            scores["volume"]["detail"].append(f"量比适中{vol_ratio:.1f}(+10)")
        elif 0.7 <= vol_ratio < 1.0:
            scores["volume"]["score"] += 7
            scores["volume"]["detail"].append(f"缩量{vol_ratio:.1f}(+7)")
        elif vol_ratio < 0.7:
            scores["volume"]["score"] += 4
            scores["volume"]["detail"].append(f"极度缩量{vol_ratio:.1f}(+4)")
        elif 2.0 < vol_ratio <= 3.0:
            scores["volume"]["score"] += 8
            scores["volume"]["detail"].append(f"放量{vol_ratio:.1f}(+8)")
        else:
            scores["volume"]["score"] += 5
            scores["volume"]["detail"].append(f"巨量{vol_ratio:.1f}(+5)")

        # 量价配合（10分）
        if "健康" in vol_signal:
            scores["volume"]["score"] += 10
            scores["volume"]["detail"].append("量价配合健康(+10)")
        elif "企稳" in vol_signal:
            scores["volume"]["score"] += 8
            scores["volume"]["detail"].append("缩量可能企稳(+8)")
        elif "警惕" in vol_signal:
            scores["volume"]["score"] += 4
            scores["volume"]["detail"].append("上涨乏力(+4)")
        elif "观望" in vol_signal:
            scores["volume"]["score"] += 2
            scores["volume"]["detail"].append("放量下跌(+2)")
        else:
            scores["volume"]["score"] += 6
            scores["volume"]["detail"].append("量价平稳(+6)")

        # 背离减分
        if divergence:
            scores["volume"]["score"] -= 3
            scores["volume"]["detail"].append(f"{divergence}(-3)")
    else:
        scores["volume"]["score"] += 10
        scores["volume"]["detail"].append("量能数据缺失(默认+10)")

    # 4. 位置分（15分）
    boll = data.get("bollinger", {})
    ma = data.get("ma", {})
    quote = data.get("quote", {})

    # 布林带位置（8分）
    boll_pos = boll.get("position", "")
    if "中轨" in boll_pos:
        scores["position"]["score"] += 6
        scores["position"]["detail"].append("布林带中轨(+6)")
    elif "下轨" in boll_pos:
        scores["position"]["score"] += 8
        scores["position"]["detail"].append("布林带下轨(+8)")
    elif "上轨" in boll_pos:
        scores["position"]["score"] += 3
        scores["position"]["detail"].append("布林带上轨(+3)")

    # 与MA20距离（7分）
    current_price = quote.get("price", 0)
    ma20 = ma.get("ma20", 0)
    if current_price > 0 and ma20 > 0:
        distance = (current_price / ma20 - 1) * 100
        if -5 <= distance <= 5:
            scores["position"]["score"] += 7
            scores["position"]["detail"].append(f"贴近MA20({distance:.1f}%)(+7)")
        elif -10 <= distance < -5:
            scores["position"]["score"] += 6
            scores["position"]["detail"].append(f"略低于MA20({distance:.1f}%)(+6)")
        elif distance < -10:
            scores["position"]["score"] += 4
            scores["position"]["detail"].append(f"远低于MA20({distance:.1f}%)(+4)")
        elif 5 < distance <= 15:
            scores["position"]["score"] += 5
            scores["position"]["detail"].append(f"略高于MA20({distance:.1f}%)(+5)")
        else:
            scores["position"]["score"] += 2
            scores["position"]["detail"].append(f"远高于MA20({distance:.1f}%)(+2)")

    # 5. 资金分（10分）
    fund = data.get("fund_flow", {})
    if fund:
        today_fund = fund.get("today", {})
        main_net = today_fund.get("main_net", 0)
        consecutive = fund.get("consecutive", {})

        if main_net > 0:
            scores["fund"]["score"] += 6
            scores["fund"]["detail"].append(f"今日主力流入({main_net:.0f}万)(+6)")
        else:
            scores["fund"]["score"] += 3
            scores["fund"]["detail"].append(f"今日主力流出({main_net:.0f}万)(+3)")

        if consecutive:
            days = consecutive.get("days", 0)
            direction = consecutive.get("direction", "")
            if direction == "流入" and days >= 3:
                scores["fund"]["score"] += 4
                scores["fund"]["detail"].append(f"连续{days}日流入(+4)")
            elif direction == "流入":
                scores["fund"]["score"] += 2
                scores["fund"]["detail"].append(f"连续{days}日流入(+2)")
    else:
        scores["fund"]["score"] += 5
        scores["fund"]["detail"].append("资金数据缺失(默认+5)")

    # 汇总
    total = sum(s["score"] for s in scores.values())

    # 评级
    if total >= 80:
        level = "强势"
        suggestion = "可积极参与"
    elif total >= 65:
        level = "偏强"
        suggestion = "可适度参与"
    elif total >= 50:
        level = "中性"
        suggestion = "观望为主"
    elif total >= 35:
        level = "偏弱"
        suggestion = "谨慎"
    else:
        level = "弱势"
        suggestion = "回避"

    return {
        "total": total,
        "level": level,
        "suggestion": suggestion,
        "breakdown": scores
    }


# ============ 主分析函数 ============

def analyze_stock(code, market="a"):
    """综合分析个股"""
    result = {
        "metadata": {
            "code": code,
            "market": market,
            "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source": "AKShare",
            "version": "2.0"
        }
    }

    # 1. 获取日线K线数据
    if market == "hk":
        df = fetch_hk_stock_kline(code)
    else:
        df = fetch_a_stock_kline(code)

    if df is None or df.empty:
        result["error"] = "无法获取K线数据"
        return result

    # 2. 生成周线数据
    log("生成周线数据...")
    df_weekly = resample_to_weekly(df)

    # 3. 基础行情
    latest = df.iloc[-1]
    result["quote"] = {
        "price": safe_float(latest['close']),
        "open": safe_float(latest['open']),
        "high": safe_float(latest['high']),
        "low": safe_float(latest['low']),
        "volume": safe_float(latest.get('volume', 0)),
        "date": latest['date'].strftime("%Y-%m-%d")
    }

    # 4. 区间涨跌幅
    result["performance"] = calc_performance(df)

    # 5. 成交量分析（新增）
    result["volume_analysis"] = calc_volume_analysis(df)

    # 6. 技术指标
    close = df['close']

    # MA
    result["ma"] = {
        "ma5": safe_float(calc_ma(close, 5).iloc[-1]),
        "ma10": safe_float(calc_ma(close, 10).iloc[-1]),
        "ma20": safe_float(calc_ma(close, 20).iloc[-1]),
        "ma60": safe_float(calc_ma(close, 60).iloc[-1]) if len(close) >= 60 else None,
        "ma120": safe_float(calc_ma(close, 120).iloc[-1]) if len(close) >= 120 else None,
    }

    # MACD
    dif, dea, macd = calc_macd(close)
    result["macd"] = {
        "dif": safe_float(dif.iloc[-1]),
        "dea": safe_float(dea.iloc[-1]),
        "macd": safe_float(macd.iloc[-1]),
        "signal": "金叉" if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2] else
                  "死叉" if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2] else
                  "多头" if dif.iloc[-1] > dea.iloc[-1] else "空头"
    }

    # RSI
    rsi = calc_rsi(close)
    rsi_val = safe_float(rsi.iloc[-1])
    result["rsi"] = {
        "value": rsi_val,
        "signal": "严重超买" if rsi_val > 80 else "超买" if rsi_val > 70 else
                  "严重超卖" if rsi_val < 20 else "超卖" if rsi_val < 30 else "中性"
    }

    # 布林带
    upper, mid, lower = calc_bollinger(close)
    current_price = close.iloc[-1]
    result["bollinger"] = {
        "upper": safe_float(upper.iloc[-1]),
        "middle": safe_float(mid.iloc[-1]),
        "lower": safe_float(lower.iloc[-1]),
        "position": "上轨附近" if current_price > mid.iloc[-1] + (upper.iloc[-1] - mid.iloc[-1]) * 0.7 else
                    "下轨附近" if current_price < mid.iloc[-1] - (mid.iloc[-1] - lower.iloc[-1]) * 0.7 else
                    "中轨附近",
        "bandwidth": safe_float((upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1] * 100)  # 带宽
    }

    # ATR（新增）
    atr = calc_atr(df)
    atr_val = safe_float(atr.iloc[-1])
    result["atr"] = {
        "value": atr_val,
        "pct": safe_float(atr_val / current_price * 100),  # ATR占价格的百分比
        "stop_loss_ref": safe_float(current_price - 2 * atr_val),  # 2倍ATR止损参考
        "stop_loss_pct": safe_float(-2 * atr_val / current_price * 100)
    }

    # 7. 多周期趋势分析（新增）
    result["multi_timeframe"] = calc_multi_timeframe_trend(df, df_weekly)

    # 8. 改进的支撑压力位
    result["levels"] = calc_support_resistance_v2(df, result["ma"])

    # 9. 资金流向（仅A股个股）
    if market == "a" and not (code.startswith('5') or code.startswith('1')):
        result["fund_flow"] = fetch_fund_flow(code)

    # 10. 财务数据（仅个股）
    if market == "a" and not (code.startswith('5') or code.startswith('1')):
        result["financial"] = fetch_financial_data(code)

    # 11. 估值数据
    if market == "a" and not (code.startswith('5') or code.startswith('1')):
        result["valuation"] = fetch_valuation(code)

    # 12. ETF特有信息
    if code.startswith('5') or code.startswith('1'):
        result["etf_info"] = fetch_etf_info(code)

    # 13. 龙虎榜（仅A股个股）
    if market == "a" and not (code.startswith('5') or code.startswith('1')):
        result["dragon_tiger"] = fetch_dragon_tiger(code)

    # 14. 综合评分（新版）
    result["score"] = calc_comprehensive_score(result)

    return result


# ============ 主函数 ============

def main():
    if len(sys.argv) < 2:
        log("用法: python3 fetch_stock_analysis.py <代码> [--market hk]")
        sys.exit(1)

    code = sys.argv[1]
    market = "a"

    if "--market" in sys.argv:
        idx = sys.argv.index("--market")
        if idx + 1 < len(sys.argv):
            market = sys.argv[idx + 1]

    # 自动判断市场
    if code.startswith('0') and len(code) == 5:
        market = "hk"

    result = analyze_stock(code, market)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
