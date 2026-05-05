# TradingAgents 技术设计文档

## 1. 项目概述

### 1.1 项目简介
TradingAgents 是一个基于多智能体（Multi-Agent）架构的金融交易框架，通过部署专门的LLM驱动的智能体，模拟真实世界交易公司的动态协作流程。该框架从基本面分析、情绪分析、技术分析，到交易执行、风险管理，实现了全流程的智能化决策。

### 1.2 核心特性
- **多智能体协作架构**：分工明确的专业化智能体团队
- **结构化输出**：Research Manager、Trader、Portfolio Manager 使用 Pydantic 结构化输出
- **持久化决策日志**：自动存储决策并在后续运行中反思学习
- **检查点恢复机制**：基于 LangGraph 的状态持久化，支持中断恢复
- **多 LLM 提供商支持**：OpenAI、Google、Anthropic、xAI、DeepSeek、Qwen、GLM、Azure、Ollama
- **五级评级系统**：Buy / Overweight / Hold / Underweight / Sell

### 1.3 技术栈
- **核心框架**：LangGraph（状态图编排）、LangChain（LLM 交互）
- **数据源**：yfinance、Alpha Vantage（股票数据、技术指标、基本面数据、新闻）
- **存储**：SQLite（检查点）、Markdown（决策日志）、JSON（状态日志）
- **开发语言**：Python 3.10+
- **依赖管理**：setuptools、pip、uv

---

## 2. 架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     TradingAgentsGraph                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Analyst Team                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │   Market     │  │   Sentiment  │  │     News     │  │   │
│  │  │   Analyst    │  │   Analyst    │  │   Analyst    │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │  ┌──────────────┐                                       │   │
│  │  │ Fundamentals │                                       │   │
│  │  │   Analyst    │                                       │   │
│  │  └──────────────┘                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Researcher Team                         │   │
│  │  ┌──────────────┐          ┌──────────────┐            │   │
│  │  │     Bull     │  ←───→   │     Bear     │            │   │
│  │  │  Researcher  │  Debate  │  Researcher  │            │   │
│  │  └──────────────┘          └──────────────┘            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Research Manager (Structured)               │   │
│  │         → Investment Plan (ResearchPlan)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                Trader (Structured)                       │   │
│  │         → Transaction Proposal (TraderProposal)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                Risk Management Team                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Aggressive   │  │   Neutral    │  │ Conservative │  │   │
│  │  │   Analyst    │  │   Analyst    │  │   Analyst    │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Portfolio Manager (Structured)                  │   │
│  │      → Final Decision (PortfolioDecision)                │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             ↓
        ┌────────────────────────────────────────┐
        │     Persistence & Memory System        │
        │  - Decision Log (Markdown)             │
        │  - Checkpoint (SQLite)                 │
        │  - State Log (JSON)                    │
        └────────────────────────────────────────┘
```

### 2.2 核心组件关系

```
TradingAgentsGraph
├── LLM Clients (deep_thinking_llm, quick_thinking_llm)
├── Tool Nodes (market, social, news, fundamentals)
├── ConditionalLogic (控制流转)
├── GraphSetup (图结构配置)
├── Propagator (状态初始化与执行)
├── Reflector (决策反思)
├── SignalProcessor (信号处理)
└── TradingMemoryLog (持久化记忆)
```

---

## 3. 智能体详细设计

### 3.1 Analyst Team（分析师团队）

#### 3.1.1 Market Analyst（市场分析师）
**职责**：分析技术指标和市场趋势

**工具集**：
- `get_stock_data`：获取股票历史价格数据
- `get_indicators`：获取技术指标（MA, MACD, RSI, Bollinger Bands, ATR, VWMA 等）

**输出**：
- 市场报告（`market_report`）
- 技术指标分析表格（Markdown 格式）

**技术指标类别**：
1. **移动平均线**：50 SMA, 200 SMA, 10 EMA
2. **MACD 系列**：macd, macds, macdh
3. **动量指标**：RSI
4. **波动率指标**：Bollinger Bands, ATR
5. **成交量指标**：VWMA

#### 3.1.2 Sentiment Analyst（情绪分析师）
**职责**：分析社交媒体和公众情绪

**工具集**：
- `get_news`：获取相关新闻用于情绪分析

**输出**：
- 情绪报告（`sentiment_report`）
- 情绪评分和市场情绪趋势

#### 3.1.3 News Analyst（新闻分析师）
**职责**：监控全球新闻和宏观经济指标

**工具集**：
- `get_news`：获取公司新闻
- `get_global_news`：获取全球新闻
- `get_insider_transactions`：获取内部交易信息

**输出**：
- 新闻报告（`news_report`）
- 事件影响分析

#### 3.1.4 Fundamentals Analyst（基本面分析师）
**职责**：评估公司财务和业绩指标

**工具集**：
- `get_fundamentals`：获取基本面数据
- `get_balance_sheet`：获取资产负债表
- `get_cashflow`：获取现金流量表
- `get_income_statement`：获取损益表

**输出**：
- 基本面报告（`fundamentals_report`）
- 财务健康度评估

### 3.2 Researcher Team（研究员团队）

#### 3.2.1 Bull Researcher（多头研究员）
**职责**：构建看涨论据，强调增长潜力和竞争优势

**输入**：
- 所有分析师报告
- 当前辩论历史
- 空头最新论据

**输出**：
- 多头论据（添加到 `investment_debate_state.bull_history`）
- 反驳空头观点的具体数据

**策略**：
- 强调增长潜力、市场机会、收入预测
- 突出竞争优势、品牌力、市场地位
- 使用财务健康度、行业趋势作为证据
- 直接参与辩论，而非简单列举数据

#### 3.2.2 Bear Researcher（空头研究员）
**职责**：构建看跌论据，识别风险和潜在问题

**输入**：
- 所有分析师报告
- 当前辩论历史
- 多头最新论据

**输出**：
- 空头论据（添加到 `investment_debate_state.bear_history`）
- 风险点和潜在红旗分析

**策略**：
- 识别估值过高、竞争威胁、监管风险
- 分析技术指标疲软信号
- 挑战多头假设的合理性

#### 3.2.3 Debate Mechanism（辩论机制）
**轮次控制**：
- 最大轮数：`max_debate_rounds`（默认 1）
- 交替发言：Bull → Bear → Bull → ...
- 终止条件：达到 `2 * max_debate_rounds` 次发言

**状态管理**：
```python
InvestDebateState:
  - history: str              # 完整辩论历史
  - bull_history: str         # 多头论据历史
  - bear_history: str         # 空头论据历史
  - current_response: str     # 最新发言
  - count: int                # 发言计数
  - judge_decision: str       # 最终裁决（由 Research Manager 生成）
```

### 3.3 Research Manager（研究经理）

#### 3.3.1 职责
综合多空辩论，生成结构化投资计划

#### 3.3.2 结构化输出（ResearchPlan）
```python
class ResearchPlan(BaseModel):
    recommendation: PortfolioRating  # Buy/Overweight/Hold/Underweight/Sell
    rationale: str                   # 关键论据总结
    strategic_actions: str           # 具体执行步骤
```

#### 3.3.3 输出示例
```markdown
**Recommendation**: Buy

**Rationale**: 多头论据在增长潜力和市场地位方面提供了更有力的证据...

**Strategic Actions**: 1) 在当前价格建仓 5% 仓位；2) 设置止损位于支撑位...
```

### 3.4 Trader（交易员）

#### 3.4.1 职责
将研究计划转化为具体交易提案

#### 3.4.2 结构化输出（TraderProposal）
```python
class TraderProposal(BaseModel):
    action: TraderAction             # Buy/Hold/Sell
    reasoning: str                   # 交易理由
    entry_price: Optional[float]     # 入场价格
    stop_loss: Optional[float]       # 止损价格
    position_sizing: Optional[str]   # 仓位规模
```

#### 3.4.3 输出示例
```markdown
**Action**: Buy

**Reasoning**: 技术指标显示突破阻力位，基本面支撑增长...

**Entry Price**: 150.25

**Stop Loss**: 145.00

**Position Sizing**: 5% of portfolio

FINAL TRANSACTION PROPOSAL: **BUY**
```

### 3.5 Risk Management Team（风险管理团队）

#### 3.5.1 三角辩论机制
**参与者**：
- **Aggressive Analyst**（激进分析师）：强调机会，主张增加风险敞口
- **Neutral Analyst**（中性分析师）：平衡观点，提供客观评估
- **Conservative Analyst**（保守分析师）：强调风险，主张降低敞口

**轮次控制**：
- 最大轮数：`max_risk_discuss_rounds`（默认 1）
- 轮流发言：Aggressive → Conservative → Neutral → Aggressive → ...
- 终止条件：达到 `3 * max_risk_discuss_rounds` 次发言

#### 3.5.2 状态管理
```python
RiskDebateState:
  - history: str                         # 完整辩论历史
  - aggressive_history: str              # 激进观点历史
  - conservative_history: str            # 保守观点历史
  - neutral_history: str                 # 中性观点历史
  - latest_speaker: str                  # 最后发言者
  - current_aggressive_response: str     # 激进最新观点
  - current_conservative_response: str   # 保守最新观点
  - current_neutral_response: str        # 中性最新观点
  - count: int                           # 发言计数
  - judge_decision: str                  # 最终决策
```

### 3.6 Portfolio Manager（投资组合经理）

#### 3.6.1 职责
综合风险分析辩论，生成最终交易决策

#### 3.6.2 结构化输出（PortfolioDecision）
```python
class PortfolioDecision(BaseModel):
    rating: PortfolioRating            # 五级评级
    executive_summary: str             # 执行摘要
    investment_thesis: str             # 投资论述
    price_target: Optional[float]      # 目标价格
    time_horizon: Optional[str]        # 持有期限
```

#### 3.6.3 记忆注入
Portfolio Manager 是唯一使用历史记忆的智能体：
- **同标的历史**：最近 5 次该股票的决策和结果
- **跨标的教训**：最近 3 次其他股票的反思总结

```python
past_context = memory_log.get_past_context(ticker, n_same=5, n_cross=3)
```

#### 3.6.4 输出示例
```markdown
**Rating**: Overweight

**Executive Summary**: 建议在 145-150 区间逐步建仓，目标 3-6 个月持有期...

**Investment Thesis**: 基于分析师团队的综合评估，技术面显示突破信号...

**Price Target**: 180.00

**Time Horizon**: 3-6 months
```

---

## 4. 数据流与状态管理

### 4.1 全局状态（AgentState）

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    company_of_interest: str
    trade_date: str
    
    # Analyst Reports
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str
    
    # Debate States
    investment_debate_state: InvestDebateState
    risk_debate_state: RiskDebateState
    
    # Decision Outputs
    investment_plan: str              # Research Manager output
    trader_investment_plan: str       # Trader output
    final_trade_decision: str         # Portfolio Manager output
    
    # Memory Context
    past_context: str                 # Injected history
```

### 4.2 数据流转

```
1. Initial State (propagate)
   ↓
2. Market Analyst → market_report
   ↓
3. Sentiment Analyst → sentiment_report
   ↓
4. News Analyst → news_report
   ↓
5. Fundamentals Analyst → fundamentals_report
   ↓
6. Bull/Bear Debate → investment_debate_state
   ↓
7. Research Manager → investment_plan
   ↓
8. Trader → trader_investment_plan
   ↓
9. Risk Debate → risk_debate_state
   ↓
10. Portfolio Manager → final_trade_decision
```

### 4.3 Tool Node 执行流程

```
Analyst Node (with tools)
  ↓
Conditional Logic: should_continue_X?
  ├── Yes → Tool Node (execute tools)
  │         ↓
  │    Return to Analyst Node (with tool results)
  │
  └── No → Msg Clear Node (clean message state)
            ↓
       Next Analyst or Researcher
```

---

## 5. 持久化与恢复机制

### 5.1 Decision Log（决策日志）

#### 5.1.1 存储路径
```
~/.tradingagents/memory/trading_memory.md
```

#### 5.1.2 条目格式
```markdown
[2026-05-10 | NVDA | Buy | pending]

DECISION:
**Rating**: Buy
**Executive Summary**: ...
**Investment Thesis**: ...

<!-- ENTRY_END -->
```

#### 5.1.3 解析与更新
**Phase A（存储决策）**：
- `store_decision()`: 在 `propagate()` 结束时自动调用
- 标记为 `pending`，等待后续解析

**Phase B（解析结果）**：
- `_resolve_pending_entries()`: 在下次同标的运行前调用
- 获取持有期回报：`_fetch_returns(ticker, date, holding_days=5)`
- 计算：`raw_return`（原始回报）、`alpha_return`（相对 SPY 的超额回报）
- 生成反思：`reflector.reflect_on_final_decision()`

**更新后格式**：
```markdown
[2026-05-10 | NVDA | Buy | +12.3% | +8.1% | 5d]

DECISION:
...

REFLECTION:
这次决策成功捕捉了突破信号，但低估了宏观风险...

<!-- ENTRY_END -->
```

#### 5.1.4 记忆注入
```python
def get_past_context(ticker: str, n_same=5, n_cross=3) -> str:
    # 返回最近 5 次同标的完整记录 + 最近 3 次跨标的反思
    pass
```

#### 5.1.5 可选轮换机制
```python
config["memory_log_max_entries"] = 100  # 限制已解析条目数量
# pending 条目永不删除
```

### 5.2 Checkpoint Resume（检查点恢复）

#### 5.2.1 启用方式
```python
config["checkpoint_enabled"] = True
```

或 CLI：
```bash
tradingagents analyze --checkpoint
```

#### 5.2.2 存储路径
```
~/.tradingagents/cache/checkpoints/<TICKER>.db
```

#### 5.2.3 工作机制
- **LangGraph SQLite Checkpointer**：每个节点执行后自动保存状态
- **Thread ID**：基于 `(ticker, trade_date)` 生成唯一标识
- **恢复逻辑**：
  - 相同 `(ticker, date)` → 从最后一个节点恢复
  - 不同 `date` → 重新开始
- **清理**：成功完成后自动调用 `clear_checkpoint()`

#### 5.2.4 使用示例
```bash
# 第一次运行（中断）
tradingagents analyze NVDA 2026-05-10 --checkpoint
# 输出：Market Analyst → Sentiment Analyst → (中断)

# 第二次运行（恢复）
tradingagents analyze NVDA 2026-05-10 --checkpoint
# 输出：Resuming from step 3 for NVDA on 2026-05-10
# 继续：News Analyst → ...
```

#### 5.2.5 清除检查点
```bash
tradingagents analyze --clear-checkpoints  # 清除所有检查点
```

### 5.3 State Log（状态日志）

#### 5.3.1 存储路径
```
~/.tradingagents/logs/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<DATE>.json
```

#### 5.3.2 内容结构
```json
{
  "company_of_interest": "NVDA",
  "trade_date": "2026-05-10",
  "market_report": "...",
  "sentiment_report": "...",
  "news_report": "...",
  "fundamentals_report": "...",
  "investment_debate_state": {
    "bull_history": "...",
    "bear_history": "...",
    "history": "...",
    "current_response": "...",
    "judge_decision": "..."
  },
  "trader_investment_decision": "...",
  "risk_debate_state": {
    "aggressive_history": "...",
    "conservative_history": "...",
    "neutral_history": "...",
    "history": "...",
    "judge_decision": "..."
  },
  "investment_plan": "...",
  "final_trade_decision": "..."
}
```

---

## 6. LLM 集成

### 6.1 双层 LLM 架构

#### 6.1.1 Quick Thinking LLM
**用途**：快速任务（分析师、研究员、风险分析师、交易员）

**配置**：
```python
config["quick_think_llm"] = "gpt-5.4-mini"
```

**特点**：
- 响应速度快
- 成本较低
- 适合中等复杂度任务

#### 6.1.2 Deep Thinking LLM
**用途**：复杂推理（Research Manager、Portfolio Manager）

**配置**：
```python
config["deep_think_llm"] = "gpt-5.4"
```

**特点**：
- 推理能力强
- 适合战略决策
- 成本较高

### 6.2 支持的提供商

#### 6.2.1 提供商列表
```python
SUPPORTED_PROVIDERS = [
    "openai",      # GPT-5.x
    "google",      # Gemini 3.x
    "anthropic",   # Claude 4.x
    "xai",         # Grok 4.x
    "deepseek",    # DeepSeek
    "qwen",        # Alibaba DashScope
    "glm",         # Zhipu GLM
    "openrouter",  # OpenRouter
    "ollama",      # Local models
    "azure"        # Azure OpenAI
]
```

#### 6.2.2 结构化输出支持
- **OpenAI / xAI**: `json_schema` mode
- **Google Gemini**: `response_schema`
- **Anthropic Claude**: Tool use (function calling)
- **其他兼容提供商**: Function calling

### 6.3 提供商特定配置

#### 6.3.1 Google
```python
config["google_thinking_level"] = "high"  # "minimal", "high"
```

#### 6.3.2 OpenAI
```python
config["openai_reasoning_effort"] = "high"  # "low", "medium", "high"
```

#### 6.3.3 Anthropic
```python
config["anthropic_effort"] = "high"  # "low", "medium", "high"
```

#### 6.3.4 Backend URL
```python
config["backend_url"] = None  # 默认 None，使用各提供商原生端点
```

---

## 7. 图编排与控制流

### 7.1 LangGraph 图结构

```python
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Market Analyst", market_analyst_node)
workflow.add_node("tools_market", tool_nodes["market"])
workflow.add_node("Msg Clear Market", msg_clear_node)
# ... 其他节点

# Define edges
workflow.add_edge(START, "Market Analyst")
workflow.add_conditional_edges(
    "Market Analyst",
    should_continue_market,
    ["tools_market", "Msg Clear Market"]
)
workflow.add_edge("tools_market", "Market Analyst")
workflow.add_edge("Msg Clear Market", "Sentiment Analyst")
# ...
workflow.add_edge("Portfolio Manager", END)
```

### 7.2 条件逻辑（ConditionalLogic）

#### 7.2.1 Analyst 控制流
```python
def should_continue_market(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools_market"
    return "Msg Clear Market"
```

#### 7.2.2 Debate 控制流
```python
def should_continue_debate(state: AgentState) -> str:
    if state["investment_debate_state"]["count"] >= 2 * max_debate_rounds:
        return "Research Manager"
    if state["investment_debate_state"]["current_response"].startswith("Bull"):
        return "Bear Researcher"
    return "Bull Researcher"
```

#### 7.2.3 Risk Analysis 控制流
```python
def should_continue_risk_analysis(state: AgentState) -> str:
    if state["risk_debate_state"]["count"] >= 3 * max_risk_discuss_rounds:
        return "Portfolio Manager"
    speaker = state["risk_debate_state"]["latest_speaker"]
    if speaker.startswith("Aggressive"):
        return "Conservative Analyst"
    if speaker.startswith("Conservative"):
        return "Neutral Analyst"
    return "Aggressive Analyst"
```

### 7.3 消息清理机制

**目的**：避免上下文窗口溢出

**实现**：
```python
def create_msg_delete():
    def msg_delete_node(state):
        # 清除 messages，保留报告
        return {"messages": []}
    return msg_delete_node
```

**时机**：
- 每个 Analyst 完成工具调用后
- 进入下一个 Analyst 前

---

## 8. 结构化输出详解

### 8.1 为什么使用结构化输出

**传统问题**：
- LLM 自由文本输出格式不一致
- 难以可靠提取关键字段
- 跨提供商行为差异大

**结构化输出优势**：
- 强制 schema 约束
- 一次调用完成，无需额外解析
- 利用各提供商原生能力
- 字段描述即输出指令

### 8.2 实现机制

#### 8.2.1 绑定 Schema
```python
from tradingagents.agents.utils.structured import bind_structured

structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")
```

#### 8.2.2 调用与渲染
```python
from tradingagents.agents.utils.structured import invoke_structured_or_freetext

final_decision = invoke_structured_or_freetext(
    structured_llm,      # 结构化 LLM
    llm,                 # 回退 LLM
    prompt,              # 提示词
    render_pm_decision,  # 渲染函数
    "Portfolio Manager"  # Agent 名称
)
```

#### 8.2.3 渲染函数
```python
def render_pm_decision(decision: PortfolioDecision) -> str:
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)
```

### 8.3 三个结构化 Agent

#### 8.3.1 Research Manager
```python
class ResearchPlan(BaseModel):
    recommendation: PortfolioRating
    rationale: str
    strategic_actions: str
```

#### 8.3.2 Trader
```python
class TraderProposal(BaseModel):
    action: TraderAction
    reasoning: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    position_sizing: Optional[str]
```

#### 8.3.3 Portfolio Manager
```python
class PortfolioDecision(BaseModel):
    rating: PortfolioRating
    executive_summary: str
    investment_thesis: str
    price_target: Optional[float]
    time_horizon: Optional[str]
```

---

## 9. 配置系统

### 9.1 默认配置（DEFAULT_CONFIG）

```python
DEFAULT_CONFIG = {
    # Paths
    "project_dir": "...",
    "results_dir": "~/.tradingagents/logs",
    "data_cache_dir": "~/.tradingagents/cache",
    "memory_log_path": "~/.tradingagents/memory/trading_memory.md",
    "memory_log_max_entries": None,  # 无限制
    
    # LLM Settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.4",
    "quick_think_llm": "gpt-5.4-mini",
    "backend_url": None,
    
    # Provider-specific
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,
    
    # Checkpoint
    "checkpoint_enabled": False,
    
    # Output
    "output_language": "English",
    
    # Debate
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    
    # Data Vendors
    "data_vendors": {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    },
    "tool_vendors": {},
}
```

### 9.2 配置覆盖

#### 9.2.1 环境变量
```bash
export TRADINGAGENTS_RESULTS_DIR=/custom/path
export TRADINGAGENTS_CACHE_DIR=/custom/cache
export TRADINGAGENTS_MEMORY_LOG_PATH=/custom/memory.md
```

#### 9.2.2 代码覆盖
```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-4.6"
config["max_debate_rounds"] = 2
config["checkpoint_enabled"] = True

ta = TradingAgentsGraph(config=config)
```

---

## 10. 数据流接口

### 10.1 抽象工具方法

**位置**：`tradingagents/agents/utils/agent_utils.py`

**设计目的**：
- 解耦数据源实现
- 支持多提供商（yfinance、Alpha Vantage）
- 统一工具接口

### 10.2 工具列表

#### 10.2.1 Core Stock APIs
```python
@tool
def get_stock_data(ticker: str, start_date: str, end_date: str) -> str:
    """获取股票历史价格数据（OHLCV）"""
    pass
```

#### 10.2.2 Technical Indicators
```python
@tool
def get_indicators(ticker: str, indicators: List[str]) -> str:
    """获取技术指标（基于已缓存的价格数据）"""
    pass
```

#### 10.2.3 Fundamental Data
```python
@tool
def get_fundamentals(ticker: str) -> str:
    """获取基本面数据（市值、PE、EPS 等）"""
    pass

@tool
def get_balance_sheet(ticker: str) -> str:
    """获取资产负债表"""
    pass

@tool
def get_cashflow(ticker: str) -> str:
    """获取现金流量表"""
    pass

@tool
def get_income_statement(ticker: str) -> str:
    """获取损益表"""
    pass
```

#### 10.2.4 News & Events
```python
@tool
def get_news(ticker: str) -> str:
    """获取公司新闻"""
    pass

@tool
def get_global_news() -> str:
    """获取全球新闻"""
    pass

@tool
def get_insider_transactions(ticker: str) -> str:
    """获取内部交易信息"""
    pass
```

### 10.3 数据源配置

#### 10.3.1 Category-level
```python
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}
```

#### 10.3.2 Tool-level (覆盖)
```python
config["tool_vendors"] = {
    "get_stock_data": "alpha_vantage",  # 仅此工具使用 Alpha Vantage
}
```

### 10.4 实现层（Dataflows）

**目录结构**：
```
tradingagents/dataflows/
├── interface.py               # 抽象接口
├── config.py                  # 配置管理
├── y_finance.py               # yfinance 实现
├── yfinance_news.py           # yfinance 新闻
├── alpha_vantage.py           # Alpha Vantage 实现
├── alpha_vantage_stock.py
├── alpha_vantage_fundamentals.py
├── alpha_vantage_indicator.py
├── alpha_vantage_news.py
└── stockstats_utils.py        # 技术指标计算
```

---

## 11. 信号处理

### 11.1 SignalProcessor

**职责**：从 Portfolio Manager 的决策中提取可执行信号

**实现**：
```python
class SignalProcessor:
    def process_signal(self, full_signal: str) -> str:
        """提取评级"""
        rating = parse_rating(full_signal)
        return rating  # "Buy", "Overweight", "Hold", "Underweight", "Sell"
```

### 11.2 评级解析

```python
def parse_rating(text: str) -> str:
    """使用启发式规则提取评级"""
    # 查找 "**Rating**: XXX" 模式
    # 返回五级评级之一
    pass
```

---

## 12. 反思机制

### 12.1 Reflector

**职责**：基于实际回报生成反思

**输入**：
- `final_decision`: 原始决策文本
- `raw_return`: 原始回报（%）
- `alpha_return`: 超额回报（相对 SPY）

**输出**：
- 一段反思文本（1 段落）

**Prompt 模板**：
```python
prompt = f"""
Reflect on this trading decision and its outcome:

DECISION:
{final_decision}

OUTCOME:
- Raw return: {raw_return:+.1%}
- Alpha (vs SPY): {alpha_return:+.1%}

Provide a one-paragraph reflection:
- What worked or didn't?
- What patterns should be noted for future decisions?
- Keep it concise and actionable.
"""
```

---

## 13. CLI 与用户交互

### 13.1 CLI 入口

**命令**：
```bash
tradingagents                # 交互式 CLI
python -m cli.main           # 替代方式
```

**功能**：
- 选择股票代码（ticker）
- 选择分析日期
- 选择 LLM 提供商
- 配置研究深度（debate rounds）
- 实时显示 Agent 进度

### 13.2 主要流程

```python
# cli/main.py
def main():
    # 1. 用户输入
    ticker = questionary.text("Ticker:").ask()
    date = questionary.text("Date (YYYY-MM-DD):").ask()
    provider = questionary.select("LLM Provider:", choices=PROVIDERS).ask()
    
    # 2. 构建配置
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider
    config["deep_think_llm"] = get_deep_model(provider)
    config["quick_think_llm"] = get_quick_model(provider)
    
    # 3. 运行
    ta = TradingAgentsGraph(debug=True, config=config)
    _, decision = ta.propagate(ticker, date)
    
    # 4. 显示结果
    print(decision)
```

---

## 14. 测试与质量保证

### 14.1 测试结构

```
tests/
├── unit/                     # 单元测试
│   ├── test_agents.py
│   ├── test_memory.py
│   └── test_config.py
├── integration/              # 集成测试
│   ├── test_graph.py
│   └── test_dataflows.py
└── smoke/                    # 冒烟测试
    └── test_structured_output.py
```

### 14.2 Pytest 配置

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "unit: fast isolated unit tests",
    "integration: tests requiring external services",
    "smoke: quick sanity-check tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

### 14.3 Fixtures

**Lazy LLM Client Import**：
```python
@pytest.fixture
def mock_llm():
    # 避免导入真实 LLM 客户端
    # 使用 Mock 对象
    pass
```

**Placeholder API Keys**：
```python
@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
```

### 14.4 诊断脚本

**结构化输出冒烟测试**：
```bash
python scripts/smoke_structured_output.py
```

**作用**：
- 测试 Research Manager、Trader、Portfolio Manager
- 验证结构化输出在所有提供商上工作
- 快速发现配置问题

---

## 15. Docker 部署

### 15.1 Dockerfile

**多阶段构建**：
```dockerfile
# Stage 1: Dev
FROM python:3.13-slim as dev
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Stage 2: Runtime
FROM python:3.13-slim
WORKDIR /app
COPY --from=dev /app .
CMD ["python", "-m", "cli.main"]
```

### 15.2 Docker Compose

```yaml
services:
  tradingagents:
    build: .
    env_file: .env
    volumes:
      - ~/.tradingagents:/root/.tradingagents
    
  tradingagents-ollama:
    build: .
    profiles: ["ollama"]
    environment:
      - llm_provider=ollama
```

### 15.3 使用方式

```bash
# 标准运行
cp .env.example .env  # 添加 API keys
docker compose run --rm tradingagents

# Ollama 本地模型
docker compose --profile ollama run --rm tradingagents-ollama
```

---

## 16. 关键设计决策

### 16.1 为什么使用 LangGraph

**优势**：
- 声明式状态图编排
- 内置检查点机制
- 条件边支持
- 与 LangChain 无缝集成

**替代方案**：
- 纯手写状态机（复杂度高）
- Crew AI（灵活性低）

### 16.2 为什么双层 LLM

**原因**：
- **成本优化**：分析任务使用便宜模型，决策任务使用强大模型
- **性能平衡**：快速迭代 + 深度推理
- **灵活配置**：用户可根据预算调整

### 16.3 为什么辩论机制

**目的**：
- 模拟真实交易公司的辩论文化
- 强制考虑多空两方观点
- 避免确认偏差
- 提升决策质量

### 16.4 为什么结构化输出

**原因**：
- LLM 自由文本不可靠
- 利用提供商原生能力
- 一次调用完成，无需正则提取
- 跨提供商一致性

### 16.5 为什么持久化记忆

**原因**：
- 从过去错误中学习
- 避免重复相同错误
- 累积领域知识
- 支持长期策略优化

---

## 17. 性能与优化

### 17.1 成本优化

**策略**：
1. **双层 LLM**：80% 任务用便宜模型
2. **Message 清理**：避免上下文窗口溢出
3. **缓存数据**：股票数据本地缓存，避免重复 API 调用
4. **结构化输出**：一次调用，无需多轮对话

### 17.2 速度优化

**策略**：
1. **并行工具调用**：LangChain ToolNode 自动并行
2. **快速模型**：分析师使用 mini 模型
3. **检查点**：避免从头重跑
4. **数据预缓存**：批量获取数据

### 17.3 可扩展性

**设计**：
- **模块化 Agent**：易于添加新分析师
- **抽象数据接口**：易于添加新数据源
- **配置驱动**：无需改代码即可调整行为
- **LangGraph 图编排**：复杂流程清晰可维护

---

## 18. 安全与合规

### 18.1 API Key 管理

**推荐**：
- 使用 `.env` 文件（不提交到 Git）
- 环境变量注入
- 支持企业级密钥管理（Azure Key Vault 等）

### 18.2 数据隐私

**措施**：
- 本地缓存数据
- 不上传用户交易数据
- 日志文件本地存储

### 18.3 免责声明

**重要**：
> TradingAgents 仅供研究和教育用途。框架输出不构成投资建议。
> 交易表现取决于多种因素，包括模型选择、温度、数据质量等。
> 用户需自行承担使用本框架的所有风险。

---

## 19. 未来扩展方向

### 19.1 短期优化
- [ ] 支持实时数据流
- [ ] 增加更多技术指标
- [ ] 优化辩论机制（动态轮次）
- [ ] 多资产组合管理

### 19.2 中期功能
- [ ] 回测引擎集成（Backtrader）
- [ ] 实盘交易接口（Alpaca、IB）
- [ ] Web UI 界面
- [ ] 社区共享决策日志

### 19.3 长期愿景
- [ ] 自适应 Agent（基于历史表现调整策略）
- [ ] 多模态输入（图表、财报 PDF）
- [ ] 分布式 Agent 部署
- [ ] 强化学习优化决策权重

---

## 20. 常见问题（FAQ）

### Q1: 如何添加新的分析师？
1. 在 `tradingagents/agents/analysts/` 创建新文件
2. 定义 `create_xxx_analyst(llm)` 函数
3. 在 `GraphSetup.setup_graph()` 中注册
4. 添加到 `selected_analysts` 列表

### Q2: 如何切换 LLM 提供商？
```python
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-4.6"
config["quick_think_llm"] = "claude-4.6-haiku"
```

### Q3: 如何增加辩论轮次？
```python
config["max_debate_rounds"] = 3
config["max_risk_discuss_rounds"] = 2
```

### Q4: 检查点占用空间过大怎么办？
```bash
tradingagents analyze --clear-checkpoints  # 清除旧检查点
```

或限制记忆日志条目：
```python
config["memory_log_max_entries"] = 100
```

### Q5: 如何使用本地模型（Ollama）？
```python
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "llama3.1:70b"
config["quick_think_llm"] = "llama3.1:8b"
```

---

## 21. 贡献指南

### 21.1 代码风格
- 遵循 PEP 8
- 使用 Type Hints
- 添加 Docstrings

### 21.2 提交流程
1. Fork 仓库
2. 创建特性分支：`git checkout -b feature/xxx`
3. 提交代码：`git commit -m "Add feature xxx"`
4. 推送分支：`git push origin feature/xxx`
5. 提交 Pull Request

### 21.3 测试要求
- 单元测试覆盖核心逻辑
- 集成测试验证端到端流程
- 添加冒烟测试验证新提供商

---

## 22. 参考资源

### 22.1 论文
- [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138)

### 22.2 文档
- [LangGraph 官方文档](https://python.langchain.com/docs/langgraph)
- [LangChain 官方文档](https://python.langchain.com/docs/)

### 22.3 社区
- [GitHub Repository](https://github.com/TauricResearch/TradingAgents)
- [Discord](https://discord.com/invite/hk9PGKShPK)
- [Twitter](https://x.com/TauricResearch)

---

## 23. 版本历史

### v0.2.4 (2026-04-25)
- ✨ 结构化输出 Agent（Research Manager、Trader、Portfolio Manager）
- ✨ LangGraph 检查点恢复
- ✨ 持久化决策日志
- ✨ DeepSeek、Qwen、GLM、Azure 支持
- 🐛 修复 Windows UTF-8 编码问题
- 🐛 修复空记忆幻觉问题

### v0.2.3 (2026-03-29)
- ✨ 多语言支持
- ✨ GPT-5.4 家族模型
- ✨ 统一模型目录
- ✨ Proxy 支持

### v0.2.2 (2026-03-XX)
- ✨ GPT-5.4/Gemini 3.1/Claude 4.6 覆盖
- ✨ 五级评级系统
- ✨ OpenAI Responses API
- ✨ Anthropic effort 控制

### v0.2.0 (2026-02-XX)
- ✨ 多提供商 LLM 支持
- 🔨 改进系统架构

---

## 24. 许可与致谢

### 24.1 许可
本项目采用 [LICENSE](LICENSE) 开源协议。

### 24.2 致谢
感谢所有贡献者、设计反馈者和 Bug 报告者，详见 [CHANGELOG.md](CHANGELOG.md)。

### 24.3 引用
如果本框架对您的研究有帮助，请引用：
```bibtex
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```

---

**文档版本**: 1.0  
**生成日期**: 2026-05-05  
**维护者**: TradingAgents Development Team

---

## 附录 A: 配置参数完整列表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `project_dir` | str | 自动检测 | 项目根目录 |
| `results_dir` | str | ~/.tradingagents/logs | 结果日志目录 |
| `data_cache_dir` | str | ~/.tradingagents/cache | 数据缓存目录 |
| `memory_log_path` | str | ~/.tradingagents/memory/trading_memory.md | 记忆日志路径 |
| `memory_log_max_entries` | int/None | None | 最大记忆条目数 |
| `llm_provider` | str | openai | LLM 提供商 |
| `deep_think_llm` | str | gpt-5.4 | 深度思考模型 |
| `quick_think_llm` | str | gpt-5.4-mini | 快速思考模型 |
| `backend_url` | str/None | None | 自定义 API 端点 |
| `google_thinking_level` | str/None | None | Google 思考级别 |
| `openai_reasoning_effort` | str/None | None | OpenAI 推理强度 |
| `anthropic_effort` | str/None | None | Anthropic 努力级别 |
| `checkpoint_enabled` | bool | False | 是否启用检查点 |
| `output_language` | str | English | 输出语言 |
| `max_debate_rounds` | int | 1 | 最大辩论轮次 |
| `max_risk_discuss_rounds` | int | 1 | 最大风险讨论轮次 |
| `max_recur_limit` | int | 100 | 最大递归限制 |
| `data_vendors` | dict | yfinance | 数据供应商配置 |
| `tool_vendors` | dict | {} | 工具级供应商覆盖 |

---

## 附录 B: Agent Prompt 模板示例

### Market Analyst Prompt
```
You are a trading assistant tasked with analyzing financial markets.
Select the most relevant indicators (up to 8) from:
- Moving Averages: 50 SMA, 200 SMA, 10 EMA
- MACD Related: macd, macds, macdh
- Momentum: RSI
- Volatility: Bollinger Bands, ATR
- Volume: VWMA

Write a detailed report with a Markdown table at the end.
Current date: {current_date}
Instrument: {instrument_context}
```

### Bull Researcher Prompt
```
You are a Bull Analyst advocating for investing in the stock.
Build a strong case emphasizing:
- Growth Potential
- Competitive Advantages
- Positive Indicators
- Counter the bear argument with specific data

Resources:
- Market report: {market_report}
- Sentiment report: {sentiment_report}
- News report: {news_report}
- Fundamentals report: {fundamentals_report}
- Last bear argument: {current_response}
```

### Portfolio Manager Prompt
```
As the Portfolio Manager, synthesize the risk debate and deliver the final decision.

Rating Scale:
- Buy: Strong conviction
- Overweight: Favorable outlook
- Hold: Maintain position
- Underweight: Reduce exposure
- Sell: Exit position

Context:
- Research plan: {research_plan}
- Trader proposal: {trader_plan}
- Past lessons: {past_context}
- Risk debate: {history}

Be decisive and ground conclusions in specific evidence.
```

---

**文档结束**
