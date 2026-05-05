# TradingAgents — Claude 项目约定

## Python 环境

**始终使用项目根目录下的虚拟环境**：

```
.venv/bin/python      # Python 3.13.6
.venv/bin/pip
```

不要使用系统 python3（macOS 自带的 3.9）、`/Users/lamperwang/.local/bin/python3.13`，或 `uv run`。

示例：
```bash
.venv/bin/python -c "import tradingagents; ..."
.venv/bin/pip install some-package
```

## 项目概况

- **框架**：多智能体 LLM 金融交易框架（LangGraph + LangChain）
- **入口**：`tradingagents/graph/trading_graph.py` → `TradingAgentsGraph`

## 数据源配置（当前）

| 类别 | Vendor |
|------|--------|
| core_stock_apis | westock |
| technical_indicators | westock |
| fundamental_data | westock |
| news_data | serper |

- **westock**：`tradingagents/dataflows/westock.py`，通过 `npx -y westock-data-skillhub@1.0.3` CLI 调用
- **serper**：`tradingagents/dataflows/serper_news.py`，需要 `SERPER_API_KEY` 环境变量
