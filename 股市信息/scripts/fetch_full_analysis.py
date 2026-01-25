"""
完整投资分析数据获取脚本 v1.0
整合宏观、行业、个股三层分析

分析框架：
1. 宏观环境（大势）- 市场周期、政策环境、资金环境
2. 行业分析（中观）- 行业景气、板块强弱、政策催化
3. 个股分析（微观）- 基本面、技术面、资金面
4. 交易策略 - 买点、仓位、止损、止盈

用法：
    python3 fetch_full_analysis.py 588000           # 完整分析ETF
    python3 fetch_full_analysis.py 002594           # 完整分析A股
    python3 fetch_full_analysis.py 00700 --market hk  # 完整分析港股
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
    print(msg, file=sys.stderr)


def safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return round(float(val), 4)
    except:
        return default


# ============ 第一层：宏观环境分析 ============

def fetch_macro_environment():
    """获取宏观环境数据"""
    log("【宏观】获取市场环境...")
    result = {
        "indices": {},
        "market_trend": {},
        "north_flow": {},
        "market_sentiment": {}
    }

    # 1. 主要指数
    try:
        df = ak.stock_zh_index_spot_em()
        target = {
            "上证指数": "sh",
            "深证成指": "sz",
            "沪深300": "hs300",
            "创业板指": "cyb",
            "科创50": "kc50",
            "中证500": "zz500"
        }
        for _, row in df.iterrows():
            name = row['名称']
            if name in target:
                result["indices"][name] = {
                    "price": safe_float(row['最新价']),
                    "change": safe_float(row['涨跌幅']),
                    "change_amount": safe_float(row['涨跌额'])
                }
    except Exception as e:
        log(f"获取指数失败: {e}")

    # 2. 判断市场趋势（基于沪深300）
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').tail(120)
            close = df['close']

            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]
            current = close.iloc[-1]

            # 判断市场周期
            if ma5 > ma20 > ma60:
                cycle = "牛市"
                cycle_score = 2
            elif ma5 < ma20 < ma60:
                cycle = "熊市"
                cycle_score = -2
            elif current > ma20:
                cycle = "震荡偏多"
                cycle_score = 1
            elif current < ma20:
                cycle = "震荡偏空"
                cycle_score = -1
            else:
                cycle = "震荡"
                cycle_score = 0

            # 计算近期涨跌
            change_5d = (current / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
            change_20d = (current / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
            change_60d = (current / close.iloc[-60] - 1) * 100 if len(close) >= 60 else 0

            result["market_trend"] = {
                "cycle": cycle,
                "cycle_score": cycle_score,
                "hs300_vs_ma20": safe_float((current / ma20 - 1) * 100),
                "hs300_change_5d": safe_float(change_5d),
                "hs300_change_20d": safe_float(change_20d),
                "hs300_change_60d": safe_float(change_60d)
            }
    except Exception as e:
        log(f"获取市场趋势失败: {e}")

    # 3. 北向资金
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if not df.empty:
            df = df.sort_values('日期').tail(20)

            today_flow = safe_float(df.iloc[-1].get('当日成交净买额', 0))

            # 近5日累计
            recent5 = df.tail(5)
            flow_5d = safe_float(recent5['当日成交净买额'].sum()) if '当日成交净买额' in recent5.columns else 0

            # 近20日累计
            flow_20d = safe_float(df['当日成交净买额'].sum()) if '当日成交净买额' in df.columns else 0

            # 连续流入/流出天数
            consecutive = 0
            direction = None
            for i in range(len(df) - 1, -1, -1):
                val = df.iloc[i].get('当日成交净买额', 0)
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

            result["north_flow"] = {
                "today": today_flow,
                "5d_total": flow_5d,
                "20d_total": flow_20d,
                "consecutive_days": consecutive,
                "direction": "流入" if direction == "in" else "流出" if direction == "out" else "无",
                "signal": "外资积极" if flow_5d > 100 else "外资撤退" if flow_5d < -100 else "外资观望"
            }
    except Exception as e:
        log(f"获取北向资金失败: {e}")

    # 4. 市场情绪（涨跌家数）
    try:
        df = ak.stock_zh_a_spot_em()
        if not df.empty:
            up_count = len(df[df['涨跌幅'] > 0])
            down_count = len(df[df['涨跌幅'] < 0])
            flat_count = len(df[df['涨跌幅'] == 0])
            total = len(df)

            up_ratio = up_count / total * 100 if total > 0 else 0

            if up_ratio > 70:
                sentiment = "极度乐观"
            elif up_ratio > 55:
                sentiment = "偏乐观"
            elif up_ratio > 45:
                sentiment = "中性"
            elif up_ratio > 30:
                sentiment = "偏悲观"
            else:
                sentiment = "极度悲观"

            result["market_sentiment"] = {
                "up_count": up_count,
                "down_count": down_count,
                "flat_count": flat_count,
                "up_ratio": safe_float(up_ratio),
                "sentiment": sentiment
            }
    except Exception as e:
        log(f"获取市场情绪失败: {e}")

    return result


# ============ 第二层：行业分析 ============

def fetch_sector_analysis(code):
    """获取行业分析数据"""
    log("【行业】获取行业数据...")
    result = {
        "sector_name": "",
        "sector_performance": {},
        "sector_rank": {},
        "sector_flow": {}
    }

    # 根据代码判断所属行业
    sector_mapping = {
        "588": "科创板",
        "515": "科技",
        "159": "创业板/科技",
        "512": "消费/周期",
        "510": "宽基",
        "516": "半导体",
        "562": "机器人",
    }

    for prefix, sector in sector_mapping.items():
        if code.startswith(prefix):
            result["sector_name"] = sector
            break

    # 1. 获取板块行情
    try:
        df = ak.stock_board_concept_name_em()
        if not df.empty:
            # 查找相关板块
            keywords = ["科创", "芯片", "半导体", "人工智能", "机器人"]
            related_sectors = []

            for _, row in df.iterrows():
                name = row.get('板块名称', '')
                for kw in keywords:
                    if kw in name:
                        related_sectors.append({
                            "name": name,
                            "change": safe_float(row.get('涨跌幅', 0)),
                            "turnover": safe_float(row.get('换手率', 0))
                        })
                        break

            # 按涨跌幅排序
            related_sectors.sort(key=lambda x: x['change'], reverse=True)
            result["related_sectors"] = related_sectors[:5]
    except Exception as e:
        log(f"获取板块行情失败: {e}")

    # 2. 获取板块资金流向
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流")
        if not df.empty:
            # 查找科技相关板块
            tech_flow = []
            keywords = ["科创", "芯片", "半导体", "人工智能", "AI", "机器人"]

            for _, row in df.iterrows():
                name = str(row.get('名称', ''))
                for kw in keywords:
                    if kw in name:
                        tech_flow.append({
                            "name": name,
                            "net_flow": safe_float(row.get('今日主力净流入-净额', 0)),
                            "net_ratio": safe_float(row.get('今日主力净流入-净占比', 0))
                        })
                        break

            result["sector_flow"] = tech_flow[:5]
    except Exception as e:
        log(f"获取板块资金流失败: {e}")

    # 3. 行业对比（ETF对比）
    try:
        comparison_etfs = [
            ("510300", "沪深300ETF"),
            ("159915", "创业板ETF"),
            ("588000", "科创50ETF"),
            ("515980", "人工智能ETF"),
            ("159995", "芯片ETF")
        ]

        df = ak.fund_etf_spot_em()
        etf_data = {str(row['代码']): row for _, row in df.iterrows()}

        comparison = []
        for etf_code, etf_name in comparison_etfs:
            if etf_code in etf_data:
                row = etf_data[etf_code]
                comparison.append({
                    "code": etf_code,
                    "name": etf_name,
                    "change": safe_float(row.get('涨跌幅', 0))
                })

        # 排序
        comparison.sort(key=lambda x: x['change'], reverse=True)
        result["etf_comparison"] = comparison

        # 判断板块强弱
        target_change = None
        hs300_change = None
        for item in comparison:
            if item["code"] == code:
                target_change = item["change"]
            if item["code"] == "510300":
                hs300_change = item["change"]

        if target_change is not None and hs300_change is not None:
            diff = target_change - hs300_change
            if diff > 1:
                result["relative_strength"] = "领涨"
            elif diff > 0:
                result["relative_strength"] = "跟涨"
            elif diff > -1:
                result["relative_strength"] = "跟跌"
            else:
                result["relative_strength"] = "领跌"
    except Exception as e:
        log(f"获取ETF对比失败: {e}")

    return result


# ============ 第三层：个股/ETF分析（导入已有脚本） ============

# 从 fetch_stock_analysis.py 导入核心函数
def calc_ma(prices, period):
    return prices.rolling(window=period).mean()

def calc_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calc_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(prices, fast)
    ema_slow = calc_ema(prices, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd = (dif - dea) * 2
    return dif, dea, macd

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calc_bollinger(prices, period=20, std_dev=2):
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    return ma + std_dev * std, ma, ma - std_dev * std


def fetch_stock_data(code, market="a"):
    """获取个股/ETF数据"""
    log(f"【个股】获取K线数据: {code}")

    try:
        if market == "hk":
            df = ak.stock_hk_hist(symbol=code, period="daily", adjust="qfq")
        elif code.startswith('5') or code.startswith('1'):
            df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
        else:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        if '日期' in df.columns:
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                                    '最高': 'high', '最低': 'low', '成交量': 'volume'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(250)
        return df
    except Exception as e:
        log(f"获取K线失败: {e}")
        return None


def analyze_technical(df):
    """技术面分析"""
    result = {}
    close = df['close']
    current = close.iloc[-1]

    # 均线
    result["ma"] = {
        "ma5": safe_float(calc_ma(close, 5).iloc[-1]),
        "ma10": safe_float(calc_ma(close, 10).iloc[-1]),
        "ma20": safe_float(calc_ma(close, 20).iloc[-1]),
        "ma60": safe_float(calc_ma(close, 60).iloc[-1]) if len(close) >= 60 else None,
    }

    # 趋势
    ma5, ma10, ma20 = result["ma"]["ma5"], result["ma"]["ma10"], result["ma"]["ma20"]
    ma60 = result["ma"]["ma60"] or ma20

    if ma5 > ma10 > ma20 > ma60:
        trend = "多头排列"
        trend_score = 2
    elif ma5 < ma10 < ma20 < ma60:
        trend = "空头排列"
        trend_score = -2
    elif current > ma20:
        trend = "偏多"
        trend_score = 1
    else:
        trend = "偏空"
        trend_score = -1

    result["trend"] = {"status": trend, "score": trend_score}

    # MACD
    dif, dea, macd = calc_macd(close)
    result["macd"] = {
        "dif": safe_float(dif.iloc[-1]),
        "dea": safe_float(dea.iloc[-1]),
        "signal": "金叉" if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2] else
                  "死叉" if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2] else
                  "多头" if dif.iloc[-1] > dea.iloc[-1] else "空头"
    }

    # RSI
    rsi_val = safe_float(calc_rsi(close).iloc[-1])
    result["rsi"] = {
        "value": rsi_val,
        "signal": "超买" if rsi_val > 70 else "超卖" if rsi_val < 30 else "中性"
    }

    # ATR止损
    atr_val = safe_float(calc_atr(df).iloc[-1])
    result["atr"] = {
        "value": atr_val,
        "stop_loss": safe_float(current - 2 * atr_val),
        "stop_loss_pct": safe_float(-2 * atr_val / current * 100)
    }

    # 量价
    if 'volume' in df.columns and len(df) >= 5:
        vol = df['volume']
        vol_ratio = vol.iloc[-1] / vol.rolling(5).mean().iloc[-1] if vol.rolling(5).mean().iloc[-1] > 0 else 1
        price_chg = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        vol_chg = (vol.iloc[-1] / vol.iloc[-2] - 1) * 100

        if price_chg > 0.5 and vol_chg > 20:
            vol_price = "放量上涨"
        elif price_chg > 0.5 and vol_chg < -20:
            vol_price = "缩量上涨"
        elif price_chg < -0.5 and vol_chg > 20:
            vol_price = "放量下跌"
        elif price_chg < -0.5 and vol_chg < -20:
            vol_price = "缩量下跌"
        else:
            vol_price = "量价平稳"

        result["volume"] = {
            "ratio": safe_float(vol_ratio),
            "vol_price": vol_price
        }

    return result


# ============ 综合评分 ============

def calc_final_score(macro, sector, technical):
    """
    综合评分
    - 宏观环境：20分
    - 行业强弱：20分
    - 技术面：60分
    """
    scores = {
        "macro": {"score": 0, "max": 20, "detail": []},
        "sector": {"score": 0, "max": 20, "detail": []},
        "technical": {"score": 0, "max": 60, "detail": []}
    }

    # 1. 宏观环境（20分）
    market_trend = macro.get("market_trend", {})
    cycle_score = market_trend.get("cycle_score", 0)

    if cycle_score == 2:
        scores["macro"]["score"] += 12
        scores["macro"]["detail"].append("牛市环境(+12)")
    elif cycle_score == 1:
        scores["macro"]["score"] += 9
        scores["macro"]["detail"].append("震荡偏多(+9)")
    elif cycle_score == 0:
        scores["macro"]["score"] += 6
        scores["macro"]["detail"].append("震荡(+6)")
    elif cycle_score == -1:
        scores["macro"]["score"] += 3
        scores["macro"]["detail"].append("震荡偏空(+3)")
    else:
        scores["macro"]["detail"].append("熊市环境(+0)")

    # 北向资金
    north = macro.get("north_flow", {})
    if north.get("direction") == "流入" and north.get("consecutive_days", 0) >= 3:
        scores["macro"]["score"] += 8
        scores["macro"]["detail"].append(f"北向连续{north['consecutive_days']}日流入(+8)")
    elif north.get("5d_total", 0) > 50:
        scores["macro"]["score"] += 5
        scores["macro"]["detail"].append("北向5日净流入(+5)")
    elif north.get("5d_total", 0) < -50:
        scores["macro"]["score"] += 2
        scores["macro"]["detail"].append("北向5日净流出(+2)")
    else:
        scores["macro"]["score"] += 4
        scores["macro"]["detail"].append("北向资金中性(+4)")

    # 2. 行业强弱（20分）
    relative = sector.get("relative_strength", "")
    if relative == "领涨":
        scores["sector"]["score"] += 15
        scores["sector"]["detail"].append("板块领涨(+15)")
    elif relative == "跟涨":
        scores["sector"]["score"] += 10
        scores["sector"]["detail"].append("板块跟涨(+10)")
    elif relative == "跟跌":
        scores["sector"]["score"] += 5
        scores["sector"]["detail"].append("板块跟跌(+5)")
    else:
        scores["sector"]["detail"].append("板块领跌(+0)")

    # 板块资金
    sector_flow = sector.get("sector_flow", [])
    if sector_flow:
        total_flow = sum(s.get("net_flow", 0) for s in sector_flow)
        if total_flow > 0:
            scores["sector"]["score"] += 5
            scores["sector"]["detail"].append("板块资金流入(+5)")
        else:
            scores["sector"]["score"] += 2
            scores["sector"]["detail"].append("板块资金流出(+2)")

    # 3. 技术面（60分）
    trend = technical.get("trend", {})
    trend_score = trend.get("score", 0)

    if trend_score == 2:
        scores["technical"]["score"] += 25
        scores["technical"]["detail"].append("多头排列(+25)")
    elif trend_score == 1:
        scores["technical"]["score"] += 18
        scores["technical"]["detail"].append("趋势偏多(+18)")
    elif trend_score == -1:
        scores["technical"]["score"] += 8
        scores["technical"]["detail"].append("趋势偏空(+8)")
    else:
        scores["technical"]["detail"].append("空头排列(+0)")

    # MACD
    macd = technical.get("macd", {})
    if macd.get("signal") == "金叉":
        scores["technical"]["score"] += 15
        scores["technical"]["detail"].append("MACD金叉(+15)")
    elif macd.get("signal") == "多头":
        scores["technical"]["score"] += 10
        scores["technical"]["detail"].append("MACD多头(+10)")
    elif macd.get("signal") == "死叉":
        scores["technical"]["score"] += 2
        scores["technical"]["detail"].append("MACD死叉(+2)")
    else:
        scores["technical"]["score"] += 5
        scores["technical"]["detail"].append("MACD空头(+5)")

    # RSI
    rsi = technical.get("rsi", {})
    rsi_val = rsi.get("value", 50)
    if 30 <= rsi_val <= 70:
        scores["technical"]["score"] += 10
        scores["technical"]["detail"].append(f"RSI中性{rsi_val:.0f}(+10)")
    elif rsi_val < 30:
        scores["technical"]["score"] += 12
        scores["technical"]["detail"].append(f"RSI超卖{rsi_val:.0f}(+12)")
    else:
        scores["technical"]["score"] += 5
        scores["technical"]["detail"].append(f"RSI超买{rsi_val:.0f}(+5)")

    # 量价
    vol = technical.get("volume", {})
    vol_price = vol.get("vol_price", "")
    if "放量上涨" in vol_price:
        scores["technical"]["score"] += 10
        scores["technical"]["detail"].append("放量上涨(+10)")
    elif "缩量下跌" in vol_price:
        scores["technical"]["score"] += 8
        scores["technical"]["detail"].append("缩量下跌(+8)")
    elif "缩量上涨" in vol_price:
        scores["technical"]["score"] += 4
        scores["technical"]["detail"].append("缩量上涨(+4)")
    else:
        scores["technical"]["score"] += 6
        scores["technical"]["detail"].append("量价平稳(+6)")

    # 汇总
    total = sum(s["score"] for s in scores.values())

    if total >= 80:
        level, suggestion = "强势", "可积极参与"
    elif total >= 65:
        level, suggestion = "偏强", "可适度参与"
    elif total >= 50:
        level, suggestion = "中性", "观望为主"
    elif total >= 35:
        level, suggestion = "偏弱", "谨慎"
    else:
        level, suggestion = "弱势", "回避"

    return {
        "total": total,
        "level": level,
        "suggestion": suggestion,
        "breakdown": scores
    }


# ============ 主函数 ============

def full_analysis(code, market="a"):
    """完整分析"""
    result = {
        "metadata": {
            "code": code,
            "market": market,
            "analyze_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "version": "1.0",
            "framework": "宏观-行业-个股"
        }
    }

    # 1. 宏观环境
    result["macro"] = fetch_macro_environment()

    # 2. 行业分析
    result["sector"] = fetch_sector_analysis(code)

    # 3. 个股/ETF技术面
    df = fetch_stock_data(code, market)
    if df is not None and not df.empty:
        result["quote"] = {
            "price": safe_float(df['close'].iloc[-1]),
            "change": safe_float((df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100) if len(df) >= 2 else 0,
            "date": df['date'].iloc[-1].strftime("%Y-%m-%d")
        }
        result["technical"] = analyze_technical(df)
    else:
        result["error"] = "无法获取K线数据"
        return result

    # 4. 综合评分
    result["score"] = calc_final_score(
        result.get("macro", {}),
        result.get("sector", {}),
        result.get("technical", {})
    )

    return result


def main():
    if len(sys.argv) < 2:
        log("用法: python3 fetch_full_analysis.py <代码> [--market hk]")
        sys.exit(1)

    code = sys.argv[1]
    market = "a"

    if "--market" in sys.argv:
        idx = sys.argv.index("--market")
        if idx + 1 < len(sys.argv):
            market = sys.argv[idx + 1]

    if code.startswith('0') and len(code) == 5:
        market = "hk"

    result = full_analysis(code, market)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
