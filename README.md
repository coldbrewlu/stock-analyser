# 股票分析系统目标与模块说明

## 一、总体目标

构建一个用于美股分析的系统，基于公开财报信息与技术指标完成以下三项分析任务：

- 公允价值估值（fundamental valuation）
- 技术分析（technical analysis）
- 回归公允价值所需时间估算（mean reversion estimation）

系统不依赖机器学习模型，仅使用公式和规则建模方法。最终通过 Docker 封装。

---

## 二、任务模块说明

### 模块 1：估值建模（Fair Value Modeling）

**输入**：
- 股票代码（ticker）
- 财报数据（通过 FMP API 提取）

**处理逻辑**：
- 使用现有估值模型（如 DCF）计算每股公允价值
- 与当前市场股价进行比较，输出是否被低估或高估（undervalued / overvalued）

**输出**：
- 每股公允价值（fair value）
- 当前价格（current price）
- 估值判断（undervalued / overvalued）

**限制条件**：
- 不使用机器学习
- 使用明确的财务公式，如 DCF 模型
- 可使用 FMP API 提供的现成估值数据作为起点

---

### 模块 2：技术分析（Technical Analysis）

**输入**：
- 股票历史价格数据（daily/weekly/monthly）

**处理逻辑**：
- 计算如下技术指标：
  - 10日移动均线（SMA 10）
  - 布林带（Bollinger Bands）
- 找出支撑位与压力位

**输出**：
- 当前价格是否接近支撑位或压力位
- 各类技术指标的当前值

---

### 模块 3：回归估算（Fair Value Convergence Time）

**输入**：
- 当前价格、公允价值
- 历史波动率
- 技术指标趋势评分

**处理逻辑**：
- 根据价格偏离度、波动性和技术信号估算回归公允价值所需的时间（单位：天）
- 以规则方式建模，不依赖机器学习

**输出**：
- 回归所需时间估计（int）

---

## 三、筛选标准（美股标的）

系统仅分析符合以下标准的美股：

- 市值大于 20 亿美元
- 在 Yahoo Finance 上有分析师目标价
- 模型计算的估值应低于至少一个分析师目标价
- 具备一定“经济护城河”评级：
  - 护城河评级 = wide / narrow / none
  - 至少为 narrow 才能继续分析
  - 可根据利润率、增长率等财务指标建立规则或评分体系判断

---

## 四、开发与部署要求

- 可通过 FMP API（免费版）提取基本估值数据（如 DCF）
- 整个系统需以模块化方式搭建
- 不要求初期准确，但需先完成功能框架
- 最终需打包为 Docker 容器，便于部署与调用


# Stock Analysis System for U.S. Equities

This project implements a rule-based stock analysis system focused on U.S. equities. It integrates fundamental valuation, technical analysis, and mean reversion estimation without relying on machine learning. The system is designed to be modular, API-accessible, and easily containerized with Docker.

## Features

- Rule-based fair value estimation using financial models (e.g., DCF)
- Technical indicator analysis (SMA, Bollinger Bands, support/resistance)
- Convergence time estimation based on volatility and signal strength
- Moat rating and company scoring based on financial health and competitive metrics
- FastAPI-powered REST API
- Asynchronous data fetching with `aiohttp` and `asyncio`

## Core Modules

1. **Valuation Engine**
   - Calculates DCF-based fair value and confidence levels
   - Implements WACC and optional relative valuation

2. **Technical Analysis Engine**
   - Calculates SMA (10, 20, 50), Bollinger Bands
   - Identifies support and resistance levels
   - Generates trend and momentum signals

3. **Convergence Estimation**
   - Predicts number of days for price to revert to fair value
   - Adjusted based on signal strength and volatility

4. **Moat Analyzer**
   - Scores company profitability, competitive strength, and financial health
   - Classifies companies as having a "Wide", "Narrow", or "None" moat

5. **Stock Screener**
   - Filters stocks based on:
     - Market cap > $2B
     - Valid analyst price targets from Yahoo Finance
     - Model fair value below at least one analyst target
     - Moat rating of at least "Narrow"

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /analysis/{symbol}` | Returns full analysis |
| `GET /valuation/{symbol}` | Returns only DCF valuation |
| `GET /technical/{symbol}` | Returns technical indicators |
| `GET /convergence/{symbol}` | Returns estimated reversion time |
| `POST /batch_analysis` | Analyze up to 10 symbols in batch |

## Sample Response (GET /analysis/AAPL)

```json
{
  "symbol": "AAPL",
  "current_price": 175.34,
  "fair_value": 192.52,
  "valuation_status": "Undervalued",
  "price_deviation": 9.81,
  "confidence_level": 91.5,
  "technical_signals": {
    "trend_signal": "bullish",
    "momentum_signal": "strong",
    "volatility_signal": "normal",
    "overall_score": 0.67
  },
  "estimated_convergence_days": 64,
  "analysis_timestamp": "2025-08-03T15:00:00"
}

## Installation 
### Run locally
```
pip install -r requirements.txt
uvicorn stock_analysis_implementation:app --reload
```


### Run with Docker
Create a `Dockerfile`:
```
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["uvicorn", "stock_analysis_implementation:app", "--host", "0.0.0.0", "--port", "8000"]
```
then build and run:
```
docker build -t stock-analyzer .
docker run -p 8000:8000 stock-analyzer

```
## Framework:
```
project/
├── main.py               # FastAPI 启动入口
├── api/
│   └── routes.py         # API 路由逻辑
├── engines/
│   ├── valuation.py      # 估值模块
│   ├── technical.py      # 技术分析模块
│   ├── convergence.py    # 回归估算模块
│   ├── moat.py           # 护城河分析模块
├── data/
│   └── fetcher.py        # 财报、价格数据抓取模块
├── models/
│   └── schemas.py        # dataclass / pydantic 数据结构
├── utils/
│   └── charts.py         # 图表绘制工具
├── screener/
│   └── screener.py       # 股票筛选逻辑
├── Dockerfile            # Docker 容器化部署脚本
├── requirements.txt      # 依赖包声明
```

## License
MIT License