"""
投资数据获取工具 - 基于 AKShare
使用前先安装: pip install akshare pandas
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

# ============ 配置区 ============
# 你的持仓代码 (示例数据，请根据实际情况修改)
HOLDINGS_A = ["510300", "510500", "510880"]
HOLDINGS_HK = ["00700"]
WATCHLIST = ["588000", "515980", "159995", "512690"]

# ============ 实时行情 ============
def get_a_stock_realtime():
    """获取A股实时行情"""
    df = ak.stock_zh_a_spot_em()
    # 筛选持仓和关注的股票
    codes = HOLDINGS_A + WATCHLIST
    df_filtered = df[df['代码'].isin(codes)]
    return df_filtered[['代码', '名称', '最新价', '涨跌幅', '成交量', '成交额', '换手率']]

def get_hk_stock_realtime():
    """获取港股实时行情"""
    df = ak.stock_hk_spot_em()
    df_filtered = df[df['代码'].isin(HOLDINGS_HK)]
    return df_filtered[['代码', '名称', '最新价', '涨跌幅']]

# ============ 历史K线 ============
def get_stock_history(code: str, days: int = 30):
    """获取个股历史K线 (前复权)"""
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    return df

# ============ 资金流向 ============
def get_fund_flow(code: str):
    """获取个股资金流向"""
    df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")
    return df.tail(10)  # 最近10天

# ============ 财务数据 ============
def get_financial_indicator(code: str):
    """获取财务指标"""
    df = ak.stock_financial_analysis_indicator(symbol=code)
    return df.head(4)  # 最近4个季度

# ============ 宏观数据 ============
def get_macro_pmi():
    """获取PMI数据"""
    df = ak.macro_china_pmi_yearly()
    return df.tail(12)  # 最近12个月

def get_macro_cpi():
    """获取CPI数据"""
    df = ak.macro_china_cpi_yearly()
    return df.tail(12)

def get_macro_gdp():
    """获取GDP数据"""
    df = ak.macro_china_gdp_yearly()
    return df.tail(8)  # 最近8个季度

# ============ 基金净值 ============
def get_fund_nav(fund_code: str):
    """获取基金历史净值"""
    df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
    return df.tail(30)  # 最近30天

# ============ 指数行情 ============
def get_index_realtime():
    """获取主要指数实时"""
    df = ak.stock_zh_index_spot_em()
    indices = ["上证指数", "深证成指", "沪深300", "中证500", "创业板指", "科创50"]
    df_filtered = df[df['名称'].isin(indices)]
    return df_filtered[['名称', '最新价', '涨跌幅']]

# ============ 研报 ============
def get_stock_research(code: str):
    """获取个股研报"""
    try:
        df = ak.stock_research_report_em(symbol=code)
        return df.head(5)  # 最近5篇
    except:
        return None

# ============ 主程序示例 ============
if __name__ == "__main__":
    print("=" * 50)
    print("主要指数")
    print("=" * 50)
    print(get_index_realtime())

    print("\n" + "=" * 50)
    print("持仓A股实时行情")
    print("=" * 50)
    print(get_a_stock_realtime())

    print("\n" + "=" * 50)
    print("最新PMI数据")
    print("=" * 50)
    print(get_macro_pmi())
