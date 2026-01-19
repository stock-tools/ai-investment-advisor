# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

这是一个 **AI 投资顾问系统** - 多模型协同投资分析平台。它结合 Claude、Codex 和 Gemini 三个 AI 模型组成"投资委员会"，通过基于技能的命令接口提供数据驱动的投资分析。

**核心理念**：数据驱动、多模型共识、个性化追踪，使用 AKShare 作为市场数据的唯一真实来源。

---

## 常用开发命令

### 运行数据获取脚本
```bash
cd "股市信息" && python3 scripts/fetch_market_data.py
```
这是所有投资分析技能的**核心数据管道**。它获取：
- 指数、持仓、关注池价格及技术指标（MA/RSI/MACD/52周位置）
- 宏观数据（PMI/CPI/M2）
- 北向资金流向
- 行业/概念板块排名及资金流
- 实时快讯（财联社电报）

**等待完成**（约2分钟）后再使用任何技能命令。脚本输出结构化 JSON 到 stdout。

### Python 依赖
```bash
pip install akshare pandas
```
核心依赖：`akshare`（市场数据）、`pandas`（数据处理）

---

## 架构设计

### 多模型投资委员会系统

系统的独特创新是**投资委员会工作流程**：

```
用户提问 → 统一输入数据（市场脚本 + 配置）
              ↓
  ┌──────────┼──────────┐
  ↓          ↓          ↓
Claude    Codex     Gemini
  ↓          ↓          ↓
独立分析 (R1)
  ↓          ↓          ↓
交叉验证 (R2，可选)
  ↓          ↓          ↓
  └──────────┼──────────┘
              ↓
       共识提取
  (3/3=强共识, 2/3=弱共识, 分歧=用户决策)
```

**关键洞察**：每个模型基于**相同数据独立分析**，然后综合各方意见提取可执行的共识。这减少了单个模型的偏见，提供更可靠的建议。

### 基于技能的命令系统

系统使用 **Claude Code 技能**（`.claude/skills/*/SKILL.md`）作为主要接口：

| 技能 | 触发词 | 用途 |
|------|--------|------|
| `/brief` | "简报", "今日市场" | 每日投资简报与持仓分析 |
| `/scan` | "有什么机会", "推荐" | 市场扫描与机会发现 |
| `/analyze` | "分析XX", "XX值得买吗" | 个股深度分析 |
| `/trade` | "买了", "卖了", "加仓" | 交易记录与价格验证 |
| `/review` | "复盘", "回顾" | 周期复盘与验证 |
| `/committee` | "开会", "投资委员会" | 多模型委员会讨论 |

**关键**：多模型工作流中，技能必须在 `.claude/skills/` 和 `~/.codex/skills/` 之间同步。

### 数据流架构

```
配置文件（用户维护）
├── Holdings.md - 持仓明细（唯一真相源）
├── Watchlist.md - 关注池及投资逻辑
├── Profile.md - 投资者自述（风格、弱点）
├── Principles.md - 投资原则
├── Context.md - 自动生成的上下文摘要
└── Insight.md - Claude 维护的用户行为洞察

        ↓

fetch_market_data.py（单一数据源）
├── 解析 Holdings.md 和 Watchlist.md
├── 从 AKShare 获取实时数据
├── 计算技术指标
└── 输出统一 JSON

        ↓

技能使用数据
├── /brief: 持仓 + 市场 → 每日简报
├── /scan: 关注池 + 市场 → 机会报告
├── /analyze: 个股 + 市场 → 深度分析
├── /trade: 用户输入 + 价格验证 → 交易记录
├── /review: 历史交易 + 当前价格 → 验证
└── /committee: 全部数据 + 多模型 → 共识
```

---

## 数据来源规则（关键）

### 优先级层级

1. **fetch_market_data.py 输出**（最高优先级）
   - 所有价格、涨跌幅、成交量必须来自脚本 JSON
   - 宏观数据（PMI/CPI/M2）来自脚本
   - 北向资金来自脚本
   - 快讯来自 `news` 字段（财联社电报）

2. **配置文件**（用户维护）
   - `Holdings.md` - 持仓、成本、数量
   - `Watchlist.md` - 关注池、投资逻辑
   - `Profile.md` - 投资风格、风险偏好

3. **WebSearch**（谨慎使用）
   - 仅用于政策文档、公司公告
   - 必须标注"来源：网络搜索"
   - 价格数据永远来自脚本，绝不使用搜索结果

### 禁止行为

- **禁止**估算或假设价格
- **禁止**使用缓存/记忆中的价格数据
- **禁止**在脚本失败时编造数据
- **禁止**在未明确标注的情况下混合数据来源

---

## 关键配置文件

### Holdings.md 结构
投资组合的"唯一真相源"，包含三个部分：

```markdown
## A股持仓
| 代码 | 名称 | 市场 | 成本价 | 持仓数量 | 市值(万) | 买入日期 |

## 港股持仓
| 代码 | 名称 | 市场 | 成本价(HKD) | 持仓数量 | 市值(万HKD) | 买入日期 |

## 基金持仓
| 代码 | 名称 | 类型 | 市场 | 成本净值 | 持有份额 | 市值(万) | 买入日期 |
```

脚本（`fetch_market_data.py` 中的 `parse_holdings_md()`）使用正则表达式解析此结构。

### Watchlist.md 结构

```markdown
## 重点关注行业
| 行业 | 关注理由 | 代表标的 | 投资逻辑 | 态度 |

## 关注的个股/ETF
| 代码 | 名称 | 市场 | 关注理由 | 核心指标 | 理想买点 | 状态 |

## 排除清单
- 项目1
- 项目2
```

### Profile.md 与 Insight.md

- **Profile.md**：用户自述的投资风格、弱点、目标
- **Insight.md**：Claude 维护的文件，追踪：
  - 实际交易中的行为模式
  - 建议采纳率
  - 决策心理观察
  - 已知弱点的改进进展

---

## 投资委员会工作流详情

多模型委员会是系统最精密的功能：

### 第一阶段：统一输入生成
1. 运行 `fetch_market_data.py` 获取市场数据
2. 读取 `Context.md` 获取持仓上下文
3. 询问用户决策问题
4. 使用 `Templates/prompt_template.md` 生成 `Committee/Input/YYYY-MM-DD-Input.md`

### 第二阶段：独立分析（R1）
每个模型（`Claude.md`、`Codex.md`、`Gemini.md`）按照 `Templates/opinion_template.md` 输出：
- 市场判断（短期/中期、风险偏好）
- 持仓信号矩阵（趋势、RSI、MACD、52周位置、估值）
- 操作建议（必须引用 ≥2 个客观数据字段）
- 风险预警
- 决策问题回答
- 自评置信度

### 第三阶段：交叉验证（R2，可选）
模型互相审阅 R1 意见并提供：
- 基于数据误读的修正
- 调整后的建议
- R2 修正表

### 第四阶段：共识提取
对比三个模型的意见：
- **强共识（3/3）**：可考虑直接执行
- **弱共识（2/3）**：参考多数意见
- **分歧**：用户自行决策

使用 `Templates/consensus_template.md` 输出到 `Committee/Sessions/YYYY-MM-DD.md`

---

## 技术实现说明

### 脚本架构（`fetch_market_data.py`）

这个 1100 行的脚本采用模块化设计，支持可切换的模块：

```python
MODULES = {
    "indices": True,      # 主要指数
    "holdings": True,     # 持仓行情
    "watchlist": True,    # 关注池
    "macro": True,        # PMI、CPI、M2
    "north_flow": True,   # 北向资金
    "sector": True,       # 行业/概念排名
    "fund_flow": True,    # 板块资金流
    "news": True,         # 财联社快讯
    "technicals": True,   # MA/RSI/MACD/52周位置
    "notices": False,     # 公告（较慢，默认关闭）
}
```

**关键函数**：
- `parse_holdings_md()`：正则解析 Holdings.md 表格
- `parse_watchlist_md()`：提取关注池和重点行业
- `enrich_with_technicals()`：为每个持仓计算技术指标
- `fetch_*_data()`：从 AKShare 获取 ETF/A股/港股/基金数据
- `compute_technicals()`：计算 MA20/60、RSI14、MACD、52周区间、波动率

### 技术指标

脚本为每个持仓计算：
- **MA20/MA60**：移动平均线及价格相对 MA 的百分比
- **RSI14**：相对强弱指标
- **MACD**：MACD 线、信号线、柱状图
- **52周位置**：当前价格在 52 周区间内的百分位
- **20日波动率**：年化波动率
- **20日动量**：20 日价格百分比变化
- **趋势**："上升"（价格>MA20>MA60）、"下降"（价格<MA20<MA60）、"震荡"（其他）

---

## 文件组织

```
ai-investment-advisor/
├── .claude/skills/           # Claude Code 技能（主接口）
│   ├── brief/SKILL.md        # 每日简报
│   ├── scan/SKILL.md         # 市场扫描
│   ├── analyze/SKILL.md      # 个股分析
│   ├── trade/SKILL.md        # 交易记录
│   ├── review/SKILL.md       # 周期复盘
│   └── committee/SKILL.md    # 多模型委员会
│
├── scripts/
│   ├── fetch_market_data.py  # 核心数据获取脚本（v2.0）
│   ├── fetch_stock_analysis.py
│   ├── fetch_full_analysis.py
│   ├── data_fetcher.py
│   └── generate_brief.py
│
├── Templates/                # 委员会模板
│   ├── prompt_template.md    # 给其他模型的输入
│   ├── opinion_template.md   # 标准观点格式
│   └── consensus_template.md # 共识汇总格式
│
├── Config-Example/           # 配置示例
│   ├── Holdings.md
│   ├── Watchlist.md
│   ├── Profile.md
│   └── Principles.md
│
└── 股市信息/                 # 用户数据目录（已 gitignore）
    ├── Config/               # 用户配置
    │   ├── Holdings.md       # 持仓唯一真相源
    │   ├── Watchlist.md      # 关注池及逻辑
    │   ├── Profile.md        # 投资者自述
    │   ├── Context.md        # 自动生成上下文
    │   └── Insight.md        # Claude 维护的洞察
    ├── Daily/                # 每日简报（YYYY-MM-DD-Brief.md）
    ├── Analysis/             # 个股分析报告
    ├── Records/              # 交易和复盘记录
    └── Committee/            # 多模型决策记录
        ├── Input/            # 统一输入数据
        ├── Opinions/         # 各模型观点
        ├── Sessions/         # 委员会会议记录
        └── Templates/        # 模板副本
```

---

## 语言要求

**所有对用户的回复必须使用中文**。这在 AGENTS.md 中有明确规定，对用户体验至关重要。

---

## 最佳实践

### 开发方面

1. **需要市场数据的技能命令前，务必先运行数据获取脚本**
2. **等待脚本执行完成** - 大约需要 2 分钟，不要着急
3. **绝不硬编码持仓数据** - 始终从 Holdings.md 读取
4. **仅使用脚本输出** - 绝不与其他数据源混合
5. **清晰标注数据来源** - "来源：AKShare" 或 "来源：网络搜索"

### 分析质量方面

1. **数据驱动而非数据罗列** - 提供分析，不只是数字
2. **结合用户实际持仓** - 针对性的建议
3. **可操作的建议** - 不是"建议关注"，而是"如果跌到 X 价位可以考虑加仓"
4. **诚实的风险评估** - 承认局限性和不确定性
5. **覆盖全部持仓** - 简报中必须分析每个持仓

### 多模型工作流方面

1. **确保输入数据完全一致** - 所有模型必须收到相同数据
2. **先进行独立分析** - R1 阶段不得互相参考
3. **结构化共识提取** - 明确的 3/3、2/3、分歧分类
4. **保持模型多样性** - 不要强行制造人工共识