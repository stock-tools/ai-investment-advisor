"""
每日投资简报生成器
基于 AKShare 获取数据，自动生成 markdown 格式简报

使用方法: python3 generate_brief.py
"""

import akshare as ak
import pandas as pd
from datetime import datetime
from pathlib import Path

# ============ 配置 ============
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "Daily"

# A股个股持仓 (示例数据，请根据实际情况修改)
HOLDINGS_STOCK = {
    # "002594": {"name": "比亚迪", "cost": 250.00, "shares": 100, "buy_date": "2024-06-01"},
}

# ETF持仓 (示例数据，请根据实际情况修改)
HOLDINGS_ETF = {
    "510300": {"name": "沪深300ETF", "cost": 3.85, "shares": 1000, "buy_date": "2024-10-10"},
    "510500": {"name": "中证500ETF", "cost": 5.50, "shares": 500, "buy_date": "2024-09-01"},
    "510880": {"name": "红利ETF", "cost": 3.20, "shares": 2000, "buy_date": "2025-01-10"},
}

# 港股持仓 (示例数据，请根据实际情况修改)
HOLDINGS_HK = {
    "00700": {"name": "腾讯控股", "cost": 350.00, "shares": 100, "buy_date": "2024-03-15"},
}

# 关注池ETF
WATCHLIST_ETF = ["588000", "515980", "159995", "512690"]


def get_index_data():
    """获取主要指数数据"""
    print("获取指数数据...")
    df = ak.stock_zh_index_spot_em()
    indices = {}
    target = ["上证指数", "深证成指", "沪深300", "中证500", "创业板指", "科创50"]
    for idx, row in df.iterrows():
        if row['名称'] in target:
            indices[row['名称']] = {
                "price": row['最新价'],
                "change": row['涨跌幅']
            }
    return indices


def get_etf_data(codes: list):
    """获取ETF实时行情"""
    print("获取ETF行情...")
    df = ak.fund_etf_spot_em()
    result = {}
    for idx, row in df.iterrows():
        code = str(row['代码'])
        if code in codes:
            result[code] = {
                "name": row['名称'],
                "price": float(row['最新价']) if pd.notna(row['最新价']) else 0,
                "change": float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
            }
    return result


def get_stock_data(codes: list):
    """获取A股个股实时行情"""
    print("获取A股个股行情...")
    df = ak.stock_zh_a_spot_em()
    result = {}
    for idx, row in df.iterrows():
        if row['代码'] in codes:
            result[row['代码']] = {
                "name": row['名称'],
                "price": float(row['最新价']) if pd.notna(row['最新价']) else 0,
                "change": float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
            }
    return result


def get_hk_stock_data(codes: list):
    """获取港股实时行情"""
    print("获取港股行情...")
    df = ak.stock_hk_spot_em()
    result = {}
    for idx, row in df.iterrows():
        if row['代码'] in codes:
            result[row['代码']] = {
                "name": row['名称'],
                "price": float(row['最新价']) if pd.notna(row['最新价']) else 0,
                "change": float(row['涨跌幅']) if pd.notna(row['涨跌幅']) else 0,
            }
    return result


def calculate_pnl(cost: float, current: float) -> float:
    """计算盈亏百分比"""
    if cost == 0:
        return 0
    return round((current - cost) / cost * 100, 2)


def calculate_holding_days(buy_date: str) -> int:
    """计算持仓天数"""
    buy = datetime.strptime(buy_date, "%Y-%m-%d")
    return (datetime.now() - buy).days


def generate_brief():
    """生成每日简报"""
    today = datetime.now().strftime("%Y-%m-%d")

    # 获取数据
    indices = get_index_data()

    # ETF数据
    etf_codes = list(HOLDINGS_ETF.keys()) + WATCHLIST_ETF
    etf_data = get_etf_data(etf_codes)

    # 个股数据
    stock_data = get_stock_data(list(HOLDINGS_STOCK.keys()))

    # 港股数据
    hk_data = get_hk_stock_data(list(HOLDINGS_HK.keys()))

    # 构建简报
    brief = f"""# 投资简报 {today}

## 一、市场概览

| 指数 | 点位 | 涨跌幅 |
|------|------|--------|
"""

    for name in ["上证指数", "沪深300", "中证500", "创业板指", "科创50"]:
        if indices.get(name):
            d = indices[name]
            sign = "+" if d['change'] >= 0 else ""
            brief += f"| {name} | {d['price']:.2f} | {sign}{d['change']:.2f}% |\n"

    # 持仓分析
    brief += """
---

## 二、持仓分析

### ETF持仓

| 代码 | 名称 | 现价 | 涨跌幅 | 成本 | 盈亏% | 持仓天数 |
|------|------|------|--------|------|-------|----------|
"""

    holdings_analysis = []

    # ETF
    for code, info in HOLDINGS_ETF.items():
        if code in etf_data:
            d = etf_data[code]
            pnl = calculate_pnl(info['cost'], d['price'])
            days = calculate_holding_days(info['buy_date'])
            sign_change = "+" if d['change'] >= 0 else ""
            sign_pnl = "+" if pnl >= 0 else ""
            brief += f"| {code} | {info['name']} | {d['price']:.4f} | {sign_change}{d['change']:.2f}% | {info['cost']} | {sign_pnl}{pnl:.2f}% | {days}天 |\n"
            holdings_analysis.append({
                "code": code,
                "name": info['name'],
                "price": d['price'],
                "change": d['change'],
                "cost": info['cost'],
                "pnl": pnl,
                "days": days,
                "type": "ETF"
            })

    # A股个股
    brief += """
### A股个股

| 代码 | 名称 | 现价 | 涨跌幅 | 成本 | 盈亏% | 持仓天数 |
|------|------|------|--------|------|-------|----------|
"""

    for code, info in HOLDINGS_STOCK.items():
        if code in stock_data:
            d = stock_data[code]
            pnl = calculate_pnl(info['cost'], d['price'])
            days = calculate_holding_days(info['buy_date'])
            sign_change = "+" if d['change'] >= 0 else ""
            sign_pnl = "+" if pnl >= 0 else ""
            brief += f"| {code} | {info['name']} | {d['price']:.2f} | {sign_change}{d['change']:.2f}% | {info['cost']} | {sign_pnl}{pnl:.2f}% | {days}天 |\n"
            holdings_analysis.append({
                "code": code,
                "name": info['name'],
                "price": d['price'],
                "change": d['change'],
                "cost": info['cost'],
                "pnl": pnl,
                "days": days,
                "type": "股票"
            })

    # 港股
    brief += """
### 港股

| 代码 | 名称 | 现价(HKD) | 涨跌幅 | 成本 | 盈亏% | 持仓天数 |
|------|------|-----------|--------|------|-------|----------|
"""

    for code, info in HOLDINGS_HK.items():
        if code in hk_data:
            d = hk_data[code]
            pnl = calculate_pnl(info['cost'], d['price'])
            days = calculate_holding_days(info['buy_date'])
            sign_change = "+" if d['change'] >= 0 else ""
            sign_pnl = "+" if pnl >= 0 else ""
            brief += f"| {code} | {info['name']} | {d['price']:.2f} | {sign_change}{d['change']:.2f}% | {info['cost']} | {sign_pnl}{pnl:.2f}% | {days}天 |\n"
            holdings_analysis.append({
                "code": code,
                "name": info['name'],
                "price": d['price'],
                "change": d['change'],
                "cost": info['cost'],
                "pnl": pnl,
                "days": days,
                "type": "港股"
            })

    # 风险预警
    brief += """
---

## 三、风险预警

"""

    risk_count = 0
    for h in holdings_analysis:
        # 深度套牢
        if h['pnl'] < -20:
            risk_count += 1
            brief += f"**{risk_count}. {h['name']}({h['code']}) - 深度套牢**\n"
            brief += f"   - 当前亏损: {h['pnl']:.2f}%，持仓 {h['days']} 天\n"
            brief += f"   - 成本: {h['cost']}，现价: {h['price']:.4f}\n"
            brief += f"   - 建议: 评估是否止损换仓\n\n"
        # 高位大幅回调
        elif h['pnl'] > 30 and h['change'] < -2:
            risk_count += 1
            brief += f"**{risk_count}. {h['name']}({h['code']}) - 高位回调**\n"
            brief += f"   - 累计盈利: +{h['pnl']:.2f}%，今日跌: {h['change']:.2f}%\n"
            brief += f"   - 建议: 考虑分批止盈保护利润\n\n"
        # 大涨后可能回调
        elif h['pnl'] > 50:
            risk_count += 1
            brief += f"**{risk_count}. {h['name']}({h['code']}) - 利润丰厚**\n"
            brief += f"   - 累计盈利: +{h['pnl']:.2f}%\n"
            brief += f"   - 建议: 关注止盈时机\n\n"

    if risk_count == 0:
        brief += "暂无高风险预警。\n\n"

    # 关注池
    brief += """---

## 四、关注池

| 代码 | 名称 | 现价 | 涨跌幅 |
|------|------|------|--------|
"""

    for code in WATCHLIST_ETF:
        if code in etf_data:
            d = etf_data[code]
            sign = "+" if d['change'] >= 0 else ""
            brief += f"| {code} | {d['name']} | {d['price']:.4f} | {sign}{d['change']:.2f}% |\n"

    # 每日复盘区域
    brief += """
---

## 五、每日复盘

**今日操作**:
- (待填写)

**市场情绪**:
- (待填写)

**明日关注**:
- (待填写)

---

*数据来源: AKShare (东方财富)*
*生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*\n"

    return brief


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    output_file = OUTPUT_DIR / f"{today}-Brief.md"

    print(f"正在生成 {today} 简报...")
    brief = generate_brief()

    # 写入文件
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(brief)

    print(f"\n简报已生成: {output_file}")
    print("\n" + "=" * 50)
    print(brief)


if __name__ == "__main__":
    main()
