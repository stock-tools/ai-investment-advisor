# 投资委员会输入文件

> 生成日期: {DATE}
> 会议模式: {MODE} - 简版/增强版

---

## 一、今日决策问题

{QUESTION}

---

## 二、市场数据快照

### 2.1 主要指数表现

{INDICES_DATA}

### 2.2 宏观指标

{MACRO_DATA}

### 2.3 资金流向

{FUND_FLOW_DATA}

---

## 三、持仓标的详细数据

{HOLDINGS_DATA}

---

## 四、关注池标的详细数据

{WATCHLIST_DATA}

---

## 五、用户画像与约束

### 5.1 基本信息
{PROFILE_DATA}

### 5.2 投资原则
{PRINCIPLES_DATA}

### 5.3 风险偏好
- 风险承受: {RISK_TOLERANCE}
- 仓位管理规则: {POSITION_RULES}

---

## 六、历史洞察

### 6.1 用户行为特征
{INSIGHT_BEHAVIOR}

### 6.2 决策采纳记录
{INSIGHT_ADOPTION}

---

## 七、数据质量清单

| 数据类别 | 完整度 | 缺失项 | 备注 |
|---------|--------|--------|------|
| 市场行情 | {MARKET_QUALITY}% | {MARKET_MISSING} |  |
| 持仓数据 | {HOLDINGS_QUALITY}% | {HOLDINGS_MISSING} |  |
| 技术指标 | {TECHNICAL_QUALITY}% | {TECHNICAL_MISSING} |  |
| 估值数据 | {VALUATION_QUALITY}% | {VALUATION_MISSING} |  |
| 资金数据 | {FLOW_QUALITY}% | {FLOW_MISSING} |  |

---

## 八、输出要求

1. **所有建议必须引用 >=2 个客观数据字段**
2. **数据缺失时必须明确标注"缺失"**
3. **关键建议必须写"关键数据依据"**
4. **置信度自评必须基于数据完整度**

---

*本输入文件由系统自动生成*