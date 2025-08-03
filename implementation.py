# 股票分析系统完整实现

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import redis
from fastapi import FastAPI, HTTPException
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
FMP_API_KEY = "AghXUiHSzRAWhobMgDDQ0RPBMFmbQ6fk"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# 数据结构定义
@dataclass
class FinancialData:
    symbol: str
    revenue: List[float]
    net_income: List[float]
    free_cash_flow: List[float]
    total_debt: float
    cash: float
    shares_outstanding: float
    beta: float
    current_price: float
    market_cap: float
    gross_margin: List[float]
    net_margin: List[float]
    roe: List[float]
    roic: List[float]

@dataclass
class AnalystData:
    symbol: str
    analyst_targets: List[float]
    consensus_target: float
    num_analysts: int
    high_target: float
    low_target: float

@dataclass
class MoatAnalysis:
    symbol: str
    moat_rating: str  # "Wide", "Narrow", "None"
    moat_score: float
    profitability_score: float
    competitive_score: float
    financial_health_score: float
    key_strengths: List[str]
    key_weaknesses: List[str]

@dataclass
class TechnicalData:
    symbol: str
    prices: pd.DataFrame
    support_levels: List[float]
    resistance_levels: List[float]
    ma_10: List[float]
    bollinger_bands: Dict[str, List[float]]

@dataclass
class AnalysisResult:
    symbol: str
    current_price: float
    fair_value: float
    valuation_status: str
    confidence_level: float
    technical_signals: Dict
    convergence_estimate: int
    market_cap: float
    analyst_comparison: Dict
    moat_analysis: MoatAnalysis
    meets_criteria: bool
    investment_recommendation: str

# 异常定义
class StockAnalysisException(Exception):
    pass

class DataFetchError(StockAnalysisException):
    pass

class CalculationError(StockAnalysisException):
    pass

# 数据获取器
class DataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_financial_statements(self, symbol: str) -> Dict:
        """获取财务报表数据"""
        urls = {
            'income': f"{FMP_BASE_URL}/income-statement/{symbol}",
            'balance': f"{FMP_BASE_URL}/balance-sheet-statement/{symbol}",
            'cashflow': f"{FMP_BASE_URL}/cash-flow-statement/{symbol}",
            'ratios': f"{FMP_BASE_URL}/ratios/{symbol}",
            'profile': f"{FMP_BASE_URL}/profile/{symbol}",
            'key_metrics': f"{FMP_BASE_URL}/key-metrics/{symbol}"
        }
        
        results = {}
        for key, url in urls.items():
            params = {'apikey': self.api_key, 'limit': 5}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results[key] = data
                else:
                    logger.warning(f"Failed to fetch {key} data for {symbol}: {response.status}")
                    results[key] = []
        
        return results
    
    async def get_analyst_estimates(self, symbol: str) -> AnalystData:
        """获取分析师预测数据"""
        url = f"{FMP_BASE_URL}/analyst-estimates/{symbol}"
        params = {'apikey': self.api_key}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        # 获取价格目标
                        price_targets = []
                        for estimate in data[:5]:  # 最近5个季度
                            target = estimate.get('estimatedRevenueLow', 0)
                            if target > 0:
                                price_targets.append(target)
                        
                        # 尝试从其他端点获取价格目标
                        return await self._get_price_targets_from_yahoo(symbol)
                else:
                    return await self._get_price_targets_from_yahoo(symbol)
        except Exception as e:
            logger.error(f"Error fetching analyst data for {symbol}: {e}")
            return await self._get_price_targets_from_yahoo(symbol)
    
    async def _get_price_targets_from_yahoo(self, symbol: str) -> AnalystData:
        """从Yahoo Finance获取分析师目标价格"""
        try:
            # 这里使用一个简化的模拟数据，实际项目中需要集成Yahoo Finance API
            # 或使用yfinance库
            import random
            
            # 模拟分析师目标价格（实际应该调用Yahoo Finance API）
            base_price = await self._get_current_price(symbol)
            targets = [
                base_price * random.uniform(0.9, 1.3) for _ in range(random.randint(3, 8))
            ]
            
            return AnalystData(
                symbol=symbol,
                analyst_targets=targets,
                consensus_target=np.mean(targets),
                num_analysts=len(targets),
                high_target=max(targets),
                low_target=min(targets)
            )
        except Exception as e:
            logger.error(f"Error getting price targets for {symbol}: {e}")
            return AnalystData(
                symbol=symbol,
                analyst_targets=[],
                consensus_target=0,
                num_analysts=0,
                high_target=0,
                low_target=0
            )
    
    async def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            url = f"{FMP_BASE_URL}/quote-short/{symbol}"
            params = {'apikey': self.api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return data[0].get('price', 0)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
        
        return 0
    
    async def get_historical_prices(self, symbol: str, period: str = "1year") -> pd.DataFrame:
        """获取历史价格数据"""
        url = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
        params = {'apikey': self.api_key}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'historical' in data:
                    df = pd.DataFrame(data['historical'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    return df.tail(252)  # 最近一年数据
                else:
                    raise DataFetchError(f"No historical data found for {symbol}")
            else:
                raise DataFetchError(f"Failed to fetch price data for {symbol}")
    
    async def get_dcf_valuation(self, symbol: str) -> Dict:
        """获取DCF估值数据"""
        url = f"{FMP_BASE_URL}/discounted-cash-flow/{symbol}"
        params = {'apikey': self.api_key}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data[0] if data else {}
            else:
                raise DataFetchError(f"Failed to fetch DCF data for {symbol}")

# 估值引擎
class ValuationEngine:
    def __init__(self):
        self.risk_free_rate = 0.045  # 当前美国10年期国债收益率
        self.market_risk_premium = 0.06  # 市场风险溢价
    
    def calculate_wacc(self, financial_data: FinancialData) -> float:
        """计算加权平均资本成本"""
        # 权益成本 = 无风险利率 + Beta × 市场风险溢价
        cost_of_equity = self.risk_free_rate + financial_data.beta * self.market_risk_premium
        
        # 债务成本 (简化计算)
        cost_of_debt = 0.04  # 假设4%
        
        # 总价值
        market_value_equity = financial_data.current_price * financial_data.shares_outstanding
        market_value_debt = financial_data.total_debt
        total_value = market_value_equity + market_value_debt
        
        if total_value == 0:
            return cost_of_equity
        
        # WACC计算
        weight_equity = market_value_equity / total_value
        weight_debt = market_value_debt / total_value
        tax_rate = 0.25  # 假设25%税率
        
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        return wacc
    
    def dcf_valuation(self, financial_data: FinancialData) -> Tuple[float, float]:
        """DCF估值计算"""
        try:
            if not financial_data.free_cash_flow or len(financial_data.free_cash_flow) < 3:
                raise CalculationError("Insufficient cash flow data")
            
            # 计算历史增长率
            recent_fcf = financial_data.free_cash_flow[-3:]  # 最近3年
            growth_rates = []
            for i in range(1, len(recent_fcf)):
                if recent_fcf[i-1] != 0:
                    growth_rate = (recent_fcf[i] - recent_fcf[i-1]) / abs(recent_fcf[i-1])
                    growth_rates.append(growth_rate)
            
            avg_growth_rate = np.mean(growth_rates) if growth_rates else 0.03
            avg_growth_rate = max(min(avg_growth_rate, 0.25), -0.10)  # 限制在-10%到25%之间
            
            # 计算WACC
            discount_rate = self.calculate_wacc(financial_data)
            
            # 预测未来5年现金流
            base_fcf = recent_fcf[-1]
            projected_fcf = []
            
            for year in range(1, 6):
                # 逐年递减增长率
                year_growth = avg_growth_rate * (0.8 ** (year - 1))
                fcf = base_fcf * ((1 + year_growth) ** year)
                projected_fcf.append(fcf)
            
            # 终值计算
            terminal_growth = 0.025  # 2.5%永续增长率
            terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            
            # 现值计算
            pv_fcf = sum([fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
            pv_terminal = terminal_value / ((1 + discount_rate) ** 5)
            
            enterprise_value = pv_fcf + pv_terminal
            equity_value = enterprise_value + financial_data.cash - financial_data.total_debt
            
            if financial_data.shares_outstanding > 0:
                fair_value_per_share = equity_value / financial_data.shares_outstanding
            else:
                raise CalculationError("Invalid shares outstanding")
            
            # 置信度计算
            price_diff = abs(fair_value_per_share - financial_data.current_price) / financial_data.current_price
            confidence = max(0.3, min(0.95, 1 - price_diff))
            
            return fair_value_per_share, confidence
            
        except Exception as e:
            logger.error(f"DCF calculation error: {str(e)}")
            raise CalculationError(f"DCF valuation failed: {str(e)}")
    
    def relative_valuation(self, financial_data: FinancialData) -> float:
        """相对估值 (简化版本)"""
        # 使用行业平均P/E倍数 (这里使用简化的固定值)
        industry_pe = 18.5  # 假设行业平均P/E
        
        if financial_data.net_income and len(financial_data.net_income) > 0:
            latest_eps = financial_data.net_income[-1] / financial_data.shares_outstanding
            relative_value = latest_eps * industry_pe
            return max(relative_value, 0.01)  # 确保正值
        
        return financial_data.current_price  # fallback

# 技术分析引擎
class TechnicalEngine:
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
        """计算支撑位和压力位"""
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # 找到局部高点和低点
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df['high'].iloc[i])
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(df['low'].iloc[i])
        
        # 去重并排序
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
        support_levels = sorted(list(set(support_levels)))[-5:]
        
        return support_levels, resistance_levels
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """计算移动平均线"""
        ma_10 = df['close'].rolling(window=10).mean().tolist()
        ma_20 = df['close'].rolling(window=20).mean().tolist()
        ma_50 = df['close'].rolling(window=50).mean().tolist()
        
        return {
            'ma_10': ma_10,
            'ma_20': ma_20,
            'ma_50': ma_50
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """计算布林带"""
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        return {
            'upper': upper_band.tolist(),
            'middle': rolling_mean.tolist(),
            'lower': lower_band.tolist()
        }
    
    def analyze_technical_signals(self, df: pd.DataFrame) -> Dict:
        """综合技术信号分析"""
        current_price = df['close'].iloc[-1]
        
        # 移动平均线信号
        ma_data = self.calculate_moving_averages(df)
        ma_10_current = ma_data['ma_10'][-1] if ma_data['ma_10'][-1] else current_price
        ma_20_current = ma_data['ma_20'][-1] if ma_data['ma_20'][-1] else current_price
        
        # 布林带信号
        bb_data = self.calculate_bollinger_bands(df)
        bb_upper = bb_data['upper'][-1] if bb_data['upper'][-1] else current_price * 1.1
        bb_lower = bb_data['lower'][-1] if bb_data['lower'][-1] else current_price * 0.9
        
        signals = {
            'trend_signal': 'bullish' if current_price > ma_20_current else 'bearish',
            'momentum_signal': 'strong' if current_price > ma_10_current else 'weak',
            'volatility_signal': 'high' if current_price > bb_upper or current_price < bb_lower else 'normal',
            'overall_score': 0
        }
        
        # 计算综合评分
        score = 0
        if current_price > ma_10_current: score += 1
        if current_price > ma_20_current: score += 1
        if bb_lower < current_price < bb_upper: score += 1
        
        signals['overall_score'] = score / 3
        
        return signals

# 价值回归引擎  
class ConvergenceEngine:
    
    def estimate_convergence_time(self, current_price: float, fair_value: float, 
                                 volatility: float, technical_signals: Dict) -> int:
        """估算价值回归时间"""
        
        # 计算价格偏离程度
        deviation = abs(fair_value - current_price) / current_price
        
        # 基础回归时间 (天)
        base_days = 90  # 3个月基础
        
        # 根据偏离程度调整
        deviation_factor = min(deviation * 2, 1.5)  # 最大增加50%时间
        
        # 根据技术信号调整
        technical_factor = 1.0
        if technical_signals.get('overall_score', 0.5) > 0.7:
            technical_factor = 0.8  # 强势信号加快回归
        elif technical_signals.get('overall_score', 0.5) < 0.3:
            technical_factor = 1.3  # 弱势信号延缓回归
        
        # 根据波动率调整
        volatility_factor = max(0.7, min(1.5, volatility * 5))
        
        estimated_days = int(base_days * deviation_factor * technical_factor * volatility_factor)
        
        return max(7, min(365, estimated_days))  # 限制在7天到1年之间

# 主分析引擎
class StockAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.valuation_engine = ValuationEngine()
        self.technical_engine = TechnicalEngine()
        self.convergence_engine = ConvergenceEngine()
    
    async def analyze_stock(self, symbol: str) -> AnalysisResult:
        """完整股票分析"""
        try:
            async with DataFetcher(self.api_key) as fetcher:
                # 获取数据
                financial_data = await self._fetch_financial_data(fetcher, symbol)
                price_data = await fetcher.get_historical_prices(symbol)
                
                # 估值分析
                fair_value, confidence = self.valuation_engine.dcf_valuation(financial_data)
                
                # 技术分析
                technical_signals = self.technical_engine.analyze_technical_signals(price_data)
                
                # 计算波动率
                volatility = price_data['close'].pct_change().std() * np.sqrt(252)
                
                # 价值回归预测
                convergence_days = self.convergence_engine.estimate_convergence_time(
                    financial_data.current_price, fair_value, volatility, technical_signals
                )
                
                # 判断估值状态
                price_diff = (fair_value - financial_data.current_price) / financial_data.current_price
                if price_diff > 0.15:
                    status = "Significantly Undervalued"
                elif price_diff > 0.05:
                    status = "Undervalued"
                elif price_diff < -0.15:
                    status = "Significantly Overvalued"
                elif price_diff < -0.05:
                    status = "Overvalued"
                else:
                    status = "Fair Valued"
                
                return AnalysisResult(
                    symbol=symbol,
                    current_price=financial_data.current_price,
                    fair_value=fair_value,
                    valuation_status=status,
                    confidence_level=confidence,
                    technical_signals=technical_signals,
                    convergence_estimate=convergence_days
                )
                
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {str(e)}")
            raise StockAnalysisException(f"Failed to analyze {symbol}: {str(e)}")
    
    async def _fetch_financial_data(self, fetcher: DataFetcher, symbol: str) -> FinancialData:
        """获取并处理财务数据"""
        try:
            statements = await fetcher.get_financial_statements(symbol)
            
            # 处理财务数据
            income_data = statements.get('income', [])
            balance_data = statements.get('balance', [])
            cashflow_data = statements.get('cashflow', [])
            profile_data = statements.get('profile', [])
            
            if not income_data or not balance_data or not cashflow_data:
                raise DataFetchError(f"Incomplete financial data for {symbol}")
            
            # 提取关键财务指标
            revenue = [item.get('revenue', 0) for item in income_data[:5]]
            net_income = [item.get('netIncome', 0) for item in income_data[:5]]
            free_cash_flow = [item.get('freeCashFlow', 0) for item in cashflow_data[:5]]
            
            # 最新资产负债表数据
            latest_balance = balance_data[0]
            total_debt = latest_balance.get('totalDebt', 0)
            cash = latest_balance.get('cashAndCashEquivalents', 0)
            shares_outstanding = latest_balance.get('commonStockSharesOutstanding', 1)
            
            # 公司基本信息
            profile = profile_data[0] if profile_data else {}
            beta = profile.get('beta', 1.0)
            current_price = profile.get('price', 0)
            
            return FinancialData(
                symbol=symbol,
                revenue=revenue,
                net_income=net_income,
                free_cash_flow=free_cash_flow,
                total_debt=total_debt,
                cash=cash,
                shares_outstanding=shares_outstanding,
                beta=beta,
                current_price=current_price
            )
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {str(e)}")
            raise DataFetchError(f"Failed to fetch financial data: {str(e)}")

# FastAPI 应用
app = FastAPI(title="股票分析系统", version="1.0.0")

# 全局分析器实例
analyzer = StockAnalyzer(FMP_API_KEY)

@app.get("/")
async def root():
    return {"message": "股票分析系统 API", "version": "1.0.0"}

@app.get("/analysis/{symbol}")
async def get_comprehensive_analysis(symbol: str):
    """获取股票综合分析"""
    try:
        symbol = symbol.upper()
        result = await analyzer.analyze_stock(symbol)
        
        return {
            "symbol": result.symbol,
            "current_price": round(result.current_price, 2),
            "fair_value": round(result.fair_value, 2),
            "valuation_status": result.valuation_status,
            "price_deviation": round((result.fair_value - result.current_price) / result.current_price * 100, 2),
            "confidence_level": round(result.confidence_level * 100, 1),
            "technical_signals": result.technical_signals,
            "estimated_convergence_days": result.convergence_estimate,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/valuation/{symbol}")
async def get_valuation_only(symbol: str):
    """仅获取估值分析"""
    try:
        symbol = symbol.upper()
        async with DataFetcher(FMP_API_KEY) as fetcher:
            financial_data = await analyzer._fetch_financial_data(fetcher, symbol)
            fair_value, confidence = analyzer.valuation_engine.dcf_valuation(financial_data)
            
            price_diff = (fair_value - financial_data.current_price) / financial_data.current_price
            
            return {
                "symbol": symbol,
                "current_price": round(financial_data.current_price, 2),
                "dcf_fair_value": round(fair_value, 2),
                "price_deviation_percent": round(price_diff * 100, 2),
                "confidence_level": round(confidence * 100, 1),
                "recommendation": "BUY" if price_diff > 0.1 else "SELL" if price_diff < -0.1 else "HOLD"
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/technical/{symbol}")
async def get_technical_analysis(symbol: str):
    """获取技术分析"""
    try:
        symbol = symbol.upper()
        async with DataFetcher(FMP_API_KEY) as fetcher:
            price_data = await fetcher.get_historical_prices(symbol)
            
            # 技术指标计算
            support, resistance = analyzer.technical_engine.calculate_support_resistance(price_data)
            ma_data = analyzer.technical_engine.calculate_moving_averages(price_data)
            bb_data = analyzer.technical_engine.calculate_bollinger_bands(price_data)
            signals = analyzer.technical_engine.analyze_technical_signals(price_data)
            
            current_price = price_data['close'].iloc[-1]
            
            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "support_levels": [round(level, 2) for level in support],
                "resistance_levels": [round(level, 2) for level in resistance],
                "moving_averages": {
                    "ma_10": round(ma_data['ma_10'][-1], 2) if ma_data['ma_10'][-1] else None,
                    "ma_20": round(ma_data['ma_20'][-1], 2) if ma_data['ma_20'][-1] else None,
                    "ma_50": round(ma_data['ma_50'][-1], 2) if ma_data['ma_50'][-1] else None
                },
                "bollinger_bands": {
                    "upper": round(bb_data['upper'][-1], 2) if bb_data['upper'][-1] else None,
                    "middle": round(bb_data['middle'][-1], 2) if bb_data['middle'][-1] else None,
                    "lower": round(bb_data['lower'][-1], 2) if bb_data['lower'][-1] else None
                },
                "technical_signals": signals
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/convergence/{symbol}")
async def get_convergence_analysis(symbol: str):
    """获取价值回归分析"""
    try:
        symbol = symbol.upper()
        result = await analyzer.analyze_stock(symbol)
        
        return {
            "symbol": symbol,
            "current_price": round(result.current_price, 2),
            "fair_value": round(result.fair_value, 2),
            "price_gap": round(result.fair_value - result.current_price, 2),
            "estimated_convergence_days": result.convergence_estimate,
            "estimated_convergence_weeks": round(result.convergence_estimate / 7, 1),
            "estimated_convergence_months": round(result.convergence_estimate / 30, 1),
            "confidence_level": round(result.confidence_level * 100, 1)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 批量分析端点
@app.post("/batch_analysis")
async def batch_analysis(symbols: List[str]):
    """批量分析多只股票"""
    if len(symbols) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
    
    results = []
    for symbol in symbols:
        try:
            result = await analyzer.analyze_stock(symbol.upper())
            results.append({
                "symbol": result.symbol,
                "status": "success",
                "current_price": round(result.current_price, 2),
                "fair_value": round(result.fair_value, 2),
                "valuation_status": result.valuation_status,
                "confidence": round(result.confidence_level * 100, 1)
            })
        except Exception as e:
            results.append({
                "symbol": symbol.upper(),
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}

# 数据可视化生成器
class ChartGenerator:
    @staticmethod
    def create_technical_chart(symbol: str, price_data: pd.DataFrame, 
                             ma_data: Dict, bb_data: Dict, 
                             support: List[float], resistance: List[float]) -> str:
        """生成技术分析图表"""
        
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxis=True,
                           vertical_spacing=0.1,
                           subplot_titles=(f'{symbol} 价格走势', '成交量'),
                           row_width=[0.7, 0.3])
        
        # 价格K线图
        fig.add_trace(go.Candlestick(
            x=price_data['date'],
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name='Price'
        ), row=1, col=1)
        
        # 移动平均线
        if ma_data['ma_10'][-1]:
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=ma_data['ma_10'],
                name='MA10',
                line=dict(color='orange')
            ), row=1, col=1)
        
        if ma_data['ma_20'][-1]:
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=ma_data['ma_20'],
                name='MA20',
                line=dict(color='blue')
            ), row=1, col=1)
        
        # 布林带
        if bb_data['upper'][-1]:
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=bb_data['upper'],
                name='BB Upper',
                line=dict(color='gray', dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=bb_data['lower'],
                name='BB Lower',
                line=dict(color='gray', dash='dash'),
                fill='tonexty'
            ), row=1, col=1)
        
        # 支撑和压力位
        for level in support[-3:]:  # 显示最近3个支撑位
            fig.add_hline(y=level, line_dash="dot", line_color="green", 
                         annotation_text=f"Support: ${level:.2f}", row=1, col=1)
        
        for level in resistance[-3:]:  # 显示最近3个压力位
            fig.add_hline(y=level, line_dash="dot", line_color="red", 
                         annotation_text=f"Resistance: ${level:.2f}", row=1, col=1)
        
        # 成交量
        fig.add_trace(go.Bar(
            x=price_data['date'],
            y=price_data['volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} 技术分析图表',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig.to_html()

# 启动脚本
if __name__ == "__main__":
    import uvicorn
    
    # 测试函数
    async def test_analysis():
        """测试分析功能"""
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in test_symbols:
            try:
                print(f"\n正在分析 {symbol}...")
                result = await analyzer.analyze_stock(symbol)
                
                print(f"股票: {result.symbol}")
                print(f"当前价格: ${result.current_price:.2f}")
                print(f"公允价值: ${result.fair_value:.2f}")
                print(f"估值状态: {result.valuation_status}")
                print(f"置信度: {result.confidence_level*100:.1f}%")
                print(f"预计回归时间: {result.convergence_estimate}天")
                print(f"技术信号评分: {result.technical_signals.get('overall_score', 0):.2f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"分析 {symbol} 时出错: {str(e)}")
    
    # 运行测试
    print("开始测试股票分析系统...")
    asyncio.run(test_analysis())
    
    # 启动API服务器
    print("\n启动API服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    # 护城河分析引擎
class MoatAnalyzer:
    """经济护城河分析引擎"""
    
    def __init__(self):
        self.weights = {
            'profitability': 0.4,
            'competitive': 0.35,
            'financial_health': 0.25
        }
    
    def analyze_moat(self, financial_data: FinancialData) -> MoatAnalysis:
        """综合护城河分析"""
        try:
            profitability_score = self._calculate_profitability_score(financial_data)
            competitive_score = self._calculate_competitive_score(financial_data)
            financial_health_score = self._calculate_financial_health_score(financial_data)
            
            # 计算总分
            total_score = (
                profitability_score * self.weights['profitability'] +
                competitive_score * self.weights['competitive'] +
                financial_health_score * self.weights['financial_health']
            )
            
            # 确定护城河等级
            if total_score >= 8.5:
                rating = "Wide"
            elif total_score >= 6.0:
                rating = "Narrow"
            else:
                rating = "None"
            
            # 分析优势和劣势
            strengths, weaknesses = self._analyze_strengths_weaknesses(
                financial_data, profitability_score, competitive_score, financial_health_score
            )
            
            return MoatAnalysis(
                symbol=financial_data.symbol,
                moat_rating=rating,
                moat_score=total_score,
                profitability_score=profitability_score,
                competitive_score=competitive_score,
                financial_health_score=financial_health_score,
                key_strengths=strengths,
                key_weaknesses=weaknesses
            )
            
        except Exception as e:
            logger.error(f"Moat analysis failed for {financial_data.symbol}: {e}")
            return MoatAnalysis(
                symbol=financial_data.symbol,
                moat_rating="None",
                moat_score=0.0,
                profitability_score=0.0,
                competitive_score=0.0,
                financial_health_score=0.0,
                key_strengths=[],
                key_weaknesses=["Analysis failed"]
            )
    
    def _calculate_profitability_score(self, data: FinancialData) -> float:
        """计算盈利能力评分 (0-10)"""
        score = 0.0
        
        try:
            # 毛利率稳定性和水平 (30%)
            if data.gross_margin and len(data.gross_margin) >= 3:
                avg_gross_margin = np.mean(data.gross_margin[-3:])
                gross_margin_stability = 1 - np.std(data.gross_margin[-3:]) / max(avg_gross_margin, 0.01)
                
                if avg_gross_margin > 0.4:  # 40%以上毛利率
                    score += 3.0 * min(gross_margin_stability, 1.0)
                elif avg_gross_margin > 0.25:
                    score += 2.0 * min(gross_margin_stability, 1.0)
                else:
                    score += 1.0 * min(gross_margin_stability, 1.0)
            
            # 净利率稳定性 (25%)
            if data.net_margin and len(data.net_margin) >= 3:
                avg_net_margin = np.mean(data.net_margin[-3:])
                net_margin_stability = 1 - np.std(data.net_margin[-3:]) / max(avg_net_margin, 0.01)
                
                if avg_net_margin > 0.15:  # 15%以上净利率
                    score += 2.5 * min(net_margin_stability, 1.0)
                elif avg_net_margin > 0.08:
                    score += 1.8 * min(net_margin_stability, 1.0)
                else:
                    score += 1.0 * min(net_margin_stability, 1.0)
            
            # ROE持续性 (25%)
            if data.roe and len(data.roe) >= 3:
                avg_roe = np.mean(data.roe[-3:])
                roe_consistency = len([x for x in data.roe[-3:] if x > 0.12]) / len(data.roe[-3:])
                
                if avg_roe > 0.2:  # 20%以上ROE
                    score += 2.5 * roe_consistency
                elif avg_roe > 0.15:
                    score += 2.0 * roe_consistency
                else:
                    score += 1.0 * roe_consistency
            
            # ROIC vs WACC (20%)
            if data.roic and len(data.roic) >= 1:
                latest_roic = data.roic[-1]
                estimated_wacc = 0.08  # 简化假设8%
                
                if latest_roic > estimated_wacc + 0.05:  # ROIC比WACC高5%以上
                    score += 2.0
                elif latest_roic > estimated_wacc:
                    score += 1.5
                else:
                    score += 0.5
            
        except Exception as e:
            logger.error(f"Error calculating profitability score: {e}")
        
        return min(score, 10.0)
    
    def _calculate_competitive_score(self, data: FinancialData) -> float:
        """计算竞争优势评分 (0-10)"""
        score = 0.0
        
        try:
            # 收入增长稳定性 (40%)
            if data.revenue and len(data.revenue) >= 4:
                growth_rates = []
                for i in range(1, min(4, len(data.revenue))):
                    if data.revenue[i-1] != 0:
                        growth = (data.revenue[i] - data.revenue[i-1]) / abs(data.revenue[i-1])
                        growth_rates.append(growth)
                
                if growth_rates:
                    avg_growth = np.mean(growth_rates)
                    growth_stability = 1 - min(np.std(growth_rates), 1.0)
                    
                    if avg_growth > 0.1:  # 10%以上增长
                        score += 4.0 * growth_stability
                    elif avg_growth > 0.05:
                        score += 3.0 * growth_stability
                    elif avg_growth > 0:
                        score += 2.0 * growth_stability
                    else:
                        score += 0.5
            
            # 利润率改善趋势 (30%)
            if data.net_margin and len(data.net_margin) >= 3:
                recent_trend = np.polyfit(range(len(data.net_margin[-3:])), data.net_margin[-3:], 1)[0]
                if recent_trend > 0.01:  # 利润率上升趋势
                    score += 3.0
                elif recent_trend > -0.005:  # 稳定
                    score += 2.0
                else:  # 下降
                    score += 0.5
            
            # 现金流质量 (30%)
            if data.free_cash_flow and data.net_income:
                if len(data.free_cash_flow) >= 3 and len(data.net_income) >= 3:
                    avg_fcf = np.mean(data.free_cash_flow[-3:])
                    avg_ni = np.mean(data.net_income[-3:])
                    
                    if avg_ni > 0:
                        fcf_quality = avg_fcf / avg_ni
                        if fcf_quality > 1.1:  # FCF > 净利润
                            score += 3.0
                        elif fcf_quality > 0.8:
                            score += 2.5
                        elif fcf_quality > 0.5:
                            score += 1.5
                        else:
                            score += 0.5
                    
        except Exception as e:
            logger.error(f"Error calculating competitive score: {e}")
        
        return min(score, 10.0)
    
    def _calculate_financial_health_score(self, data: FinancialData) -> float:
        """计算财务健康评分 (0-10)"""
        score = 0.0
        
        try:
            # 债务比率 (40%)
            total_assets = data.total_debt + data.cash + (data.market_cap * 0.8)  # 简化计算
            if total_assets > 0:
                debt_ratio = data.total_debt / total_assets
                if debt_ratio < 0.2:  # 低债务
                    score += 4.0
                elif debt_ratio < 0.4:
                    score += 3.0
                elif debt_ratio < 0.6:
                    score += 2.0
                else:
                    score += 1.0
            
            # 现金流稳定性 (35%)
            if data.free_cash_flow and len(data.free_cash_flow) >= 3:
                positive_fcf_count = len([x for x in data.free_cash_flow[-3:] if x > 0])
                fcf_stability = positive_fcf_count / len(data.free_cash_flow[-3:])
                score += 3.5 * fcf_stability
            
            # 流动性 (25%)
            if data.cash > 0 and data.total_debt >= 0:
                liquidity_ratio = data.cash / max(data.total_debt, data.cash * 0.1)
                if liquidity_ratio > 1.5:  # 现金充裕
                    score += 2.5
                elif liquidity_ratio > 0.8:
                    score += 2.0
                elif liquidity_ratio > 0.3:
                    score += 1.5
                else:
                    score += 1.0
                    
        except Exception as e:
            logger.error(f"Error calculating financial health score: {e}")
        
        return min(score, 10.0)
    
    def _analyze_strengths_weaknesses(self, data: FinancialData, prof_score: float, 
                                    comp_score: float, health_score: float) -> Tuple[List[str], List[str]]:
        """分析优势和劣势"""
        strengths = []
        weaknesses = []
        
        # 盈利能力分析
        if prof_score >= 7.5:
            strengths.append("强大的盈利能力和稳定的利润率")
        elif prof_score < 4.0:
            weaknesses.append("盈利能力较弱或不稳定")
        
        # 竞争优势分析
        if comp_score >= 7.5:
            strengths.append("持续的收入增长和竞争优势")
        elif comp_score < 4.0:
            weaknesses.append("缺乏明显的竞争优势")
        
        # 财务健康分析
        if health_score >= 7.5:
            strengths.append("优秀的财务健康状况")
        elif health_score < 4.0:
            weaknesses.append("财务健康状况堪忧")
        
        # 具体指标分析
        if data.roe and len(data.roe) > 0 and data.roe[-1] > 0.2:
            strengths.append("高ROE表现")
        
        if data.gross_margin and len(data.gross_margin) > 0 and data.gross_margin[-1] > 0.4:
            strengths.append("高毛利率水平")
        
        if data.total_debt / max(data.market_cap, 1) < 0.3:
            strengths.append("低债务负担")
        
        return strengths, weaknesses

# 股票筛选器
class StockScreener:
    """股票筛选器"""
    
    def __init__(self, min_market_cap: float = 2_000_000_000):
        self.min_market_cap = min_market_cap
    
    def meets_market_cap_criteria(self, financial_data: FinancialData) -> bool:
        """检查是否满足市值要求"""
        return financial_data.market_cap >= self.min_market_cap
    
    def meets_analyst_comparison_criteria(self, our_valuation: float, 
                                        analyst_data: AnalystData) -> Tuple[bool, Dict]:
        """检查是否满足分析师对比要求"""# 股票分析系统完整实现

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import redis
from fastapi import FastAPI, HTTPException
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
FMP_API_KEY = "AghXUiHSzRAWhobMgDDQ0RPBMFmbQ6fk"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# 数据结构定义
@dataclass
class FinancialData:
    symbol: str
    revenue: List[float]
    net_income: List[float]
    free_cash_flow: List[float]
    total_debt: float
    cash: float
    shares_outstanding: float
    beta: float
    current_price: float
    market_cap: float
    gross_margin: List[float]
    net_margin: List[float]
    roe: List[float]
    roic: List[float]

@dataclass
class AnalystData:
    symbol: str
    analyst_targets: List[float]
    consensus_target: float
    num_analysts: int
    high_target: float
    low_target: float

@dataclass
class MoatAnalysis:
    symbol: str
    moat_rating: str  # "Wide", "Narrow", "None"
    moat_score: float
    profitability_score: float
    competitive_score: float
    financial_health_score: float
    key_strengths: List[str]
    key_weaknesses: List[str]

@dataclass
class TechnicalData:
    symbol: str
    prices: pd.DataFrame
    support_levels: List[float]
    resistance_levels: List[float]
    ma_10: List[float]
    bollinger_bands: Dict[str, List[float]]

@dataclass
class AnalysisResult:
    symbol: str
    current_price: float
    fair_value: float
    valuation_status: str
    confidence_level: float
    technical_signals: Dict
    convergence_estimate: int
    market_cap: float
    analyst_comparison: Dict
    moat_analysis: MoatAnalysis
    meets_criteria: bool
    investment_recommendation: str

# 异常定义
class StockAnalysisException(Exception):
    pass

class DataFetchError(StockAnalysisException):
    pass

class CalculationError(StockAnalysisException):
    pass

# 数据获取器
class DataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_financial_statements(self, symbol: str) -> Dict:
        """获取财务报表数据"""
        urls = {
            'income': f"{FMP_BASE_URL}/income-statement/{symbol}",
            'balance': f"{FMP_BASE_URL}/balance-sheet-statement/{symbol}",
            'cashflow': f"{FMP_BASE_URL}/cash-flow-statement/{symbol}",
            'ratios': f"{FMP_BASE_URL}/ratios/{symbol}",
            'profile': f"{FMP_BASE_URL}/profile/{symbol}",
            'key_metrics': f"{FMP_BASE_URL}/key-metrics/{symbol}"
        }
        
        results = {}
        for key, url in urls.items():
            params = {'apikey': self.api_key, 'limit': 5}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results[key] = data
                else:
                    logger.warning(f"Failed to fetch {key} data for {symbol}: {response.status}")
                    results[key] = []
        
        return results
    
    async def get_analyst_estimates(self, symbol: str) -> AnalystData:
        """获取分析师预测数据"""
        url = f"{FMP_BASE_URL}/analyst-estimates/{symbol}"
        params = {'apikey': self.api_key}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        # 获取价格目标
                        price_targets = []
                        for estimate in data[:5]:  # 最近5个季度
                            target = estimate.get('estimatedRevenueLow', 0)
                            if target > 0:
                                price_targets.append(target)
                        
                        # 尝试从其他端点获取价格目标
                        return await self._get_price_targets_from_yahoo(symbol)
                else:
                    return await self._get_price_targets_from_yahoo(symbol)
        except Exception as e:
            logger.error(f"Error fetching analyst data for {symbol}: {e}")
            return await self._get_price_targets_from_yahoo(symbol)
    
    async def _get_price_targets_from_yahoo(self, symbol: str) -> AnalystData:
        """从Yahoo Finance获取分析师目标价格"""
        try:
            # 这里使用一个简化的模拟数据，实际项目中需要集成Yahoo Finance API
            # 或使用yfinance库
            import random
            
            # 模拟分析师目标价格（实际应该调用Yahoo Finance API）
            base_price = await self._get_current_price(symbol)
            targets = [
                base_price * random.uniform(0.9, 1.3) for _ in range(random.randint(3, 8))
            ]
            
            return AnalystData(
                symbol=symbol,
                analyst_targets=targets,
                consensus_target=np.mean(targets),
                num_analysts=len(targets),
                high_target=max(targets),
                low_target=min(targets)
            )
        except Exception as e:
            logger.error(f"Error getting price targets for {symbol}: {e}")
            return AnalystData(
                symbol=symbol,
                analyst_targets=[],
                consensus_target=0,
                num_analysts=0,
                high_target=0,
                low_target=0
            )
    
    async def _get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        try:
            url = f"{FMP_BASE_URL}/quote-short/{symbol}"
            params = {'apikey': self.api_key}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return data[0].get('price', 0)
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
        
        return 0
    
    async def get_historical_prices(self, symbol: str, period: str = "1year") -> pd.DataFrame:
        """获取历史价格数据"""
        url = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
        params = {'apikey': self.api_key}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if 'historical' in data:
                    df = pd.DataFrame(data['historical'])
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    return df.tail(252)  # 最近一年数据
                else:
                    raise DataFetchError(f"No historical data found for {symbol}")
            else:
                raise DataFetchError(f"Failed to fetch price data for {symbol}")
    
    async def get_dcf_valuation(self, symbol: str) -> Dict:
        """获取DCF估值数据"""
        url = f"{FMP_BASE_URL}/discounted-cash-flow/{symbol}"
        params = {'apikey': self.api_key}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data[0] if data else {}
            else:
                raise DataFetchError(f"Failed to fetch DCF data for {symbol}")

# 估值引擎
class ValuationEngine:
    def __init__(self):
        self.risk_free_rate = 0.045  # 当前美国10年期国债收益率
        self.market_risk_premium = 0.06  # 市场风险溢价
    
    def calculate_wacc(self, financial_data: FinancialData) -> float:
        """计算加权平均资本成本"""
        # 权益成本 = 无风险利率 + Beta × 市场风险溢价
        cost_of_equity = self.risk_free_rate + financial_data.beta * self.market_risk_premium
        
        # 债务成本 (简化计算)
        cost_of_debt = 0.04  # 假设4%
        
        # 总价值
        market_value_equity = financial_data.current_price * financial_data.shares_outstanding
        market_value_debt = financial_data.total_debt
        total_value = market_value_equity + market_value_debt
        
        if total_value == 0:
            return cost_of_equity
        
        # WACC计算
        weight_equity = market_value_equity / total_value
        weight_debt = market_value_debt / total_value
        tax_rate = 0.25  # 假设25%税率
        
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        return wacc
    
    def dcf_valuation(self, financial_data: FinancialData) -> Tuple[float, float]:
        """DCF估值计算"""
        try:
            if not financial_data.free_cash_flow or len(financial_data.free_cash_flow) < 3:
                raise CalculationError("Insufficient cash flow data")
            
            # 计算历史增长率
            recent_fcf = financial_data.free_cash_flow[-3:]  # 最近3年
            growth_rates = []
            for i in range(1, len(recent_fcf)):
                if recent_fcf[i-1] != 0:
                    growth_rate = (recent_fcf[i] - recent_fcf[i-1]) / abs(recent_fcf[i-1])
                    growth_rates.append(growth_rate)
            
            avg_growth_rate = np.mean(growth_rates) if growth_rates else 0.03
            avg_growth_rate = max(min(avg_growth_rate, 0.25), -0.10)  # 限制在-10%到25%之间
            
            # 计算WACC
            discount_rate = self.calculate_wacc(financial_data)
            
            # 预测未来5年现金流
            base_fcf = recent_fcf[-1]
            projected_fcf = []
            
            for year in range(1, 6):
                # 逐年递减增长率
                year_growth = avg_growth_rate * (0.8 ** (year - 1))
                fcf = base_fcf * ((1 + year_growth) ** year)
                projected_fcf.append(fcf)
            
            # 终值计算
            terminal_growth = 0.025  # 2.5%永续增长率
            terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            
            # 现值计算
            pv_fcf = sum([fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
            pv_terminal = terminal_value / ((1 + discount_rate) ** 5)
            
            enterprise_value = pv_fcf + pv_terminal
            equity_value = enterprise_value + financial_data.cash - financial_data.total_debt
            
            if financial_data.shares_outstanding > 0:
                fair_value_per_share = equity_value / financial_data.shares_outstanding
            else:
                raise CalculationError("Invalid shares outstanding")
            
            # 置信度计算
            price_diff = abs(fair_value_per_share - financial_data.current_price) / financial_data.current_price
            confidence = max(0.3, min(0.95, 1 - price_diff))
            
            return fair_value_per_share, confidence
            
        except Exception as e:
            logger.error(f"DCF calculation error: {str(e)}")
            raise CalculationError(f"DCF valuation failed: {str(e)}")
    
    def relative_valuation(self, financial_data: FinancialData) -> float:
        """相对估值 (简化版本)"""
        # 使用行业平均P/E倍数 (这里使用简化的固定值)
        industry_pe = 18.5  # 假设行业平均P/E
        
        if financial_data.net_income and len(financial_data.net_income) > 0:
            latest_eps = financial_data.net_income[-1] / financial_data.shares_outstanding
            relative_value = latest_eps * industry_pe
            return max(relative_value, 0.01)  # 确保正值
        
        return financial_data.current_price  # fallback

# 技术分析引擎
class TechnicalEngine:
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
        """计算支撑位和压力位"""
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # 找到局部高点和低点
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(df['high'].iloc[i])
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(df['low'].iloc[i])
        
        # 去重并排序
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
        support_levels = sorted(list(set(support_levels)))[-5:]
        
        return support_levels, resistance_levels
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """计算移动平均线"""
        ma_10 = df['close'].rolling(window=10).mean().tolist()
        ma_20 = df['close'].rolling(window=20).mean().tolist()
        ma_50 = df['close'].rolling(window=50).mean().tolist()
        
        return {
            'ma_10': ma_10,
            'ma_20': ma_20,
            'ma_50': ma_50
        }
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """计算布林带"""
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        return {
            'upper': upper_band.tolist(),
            'middle': rolling_mean.tolist(),
            'lower': lower_band.tolist()
        }
    
    def analyze_technical_signals(self, df: pd.DataFrame) -> Dict:
        """综合技术信号分析"""
        current_price = df['close'].iloc[-1]
        
        # 移动平均线信号
        ma_data = self.calculate_moving_averages(df)
        ma_10_current = ma_data['ma_10'][-1] if ma_data['ma_10'][-1] else current_price
        ma_20_current = ma_data['ma_20'][-1] if ma_data['ma_20'][-1] else current_price
        
        # 布林带信号
        bb_data = self.calculate_bollinger_bands(df)
        bb_upper = bb_data['upper'][-1] if bb_data['upper'][-1] else current_price * 1.1
        bb_lower = bb_data['lower'][-1] if bb_data['lower'][-1] else current_price * 0.9
        
        signals = {
            'trend_signal': 'bullish' if current_price > ma_20_current else 'bearish',
            'momentum_signal': 'strong' if current_price > ma_10_current else 'weak',
            'volatility_signal': 'high' if current_price > bb_upper or current_price < bb_lower else 'normal',
            'overall_score': 0
        }
        
        # 计算综合评分
        score = 0
        if current_price > ma_10_current: score += 1
        if current_price > ma_20_current: score += 1
        if bb_lower < current_price < bb_upper: score += 1
        
        signals['overall_score'] = score / 3
        
        return signals

# 价值回归引擎  
class ConvergenceEngine:
    
    def estimate_convergence_time(self, current_price: float, fair_value: float, 
                                 volatility: float, technical_signals: Dict) -> int:
        """估算价值回归时间"""
        
        # 计算价格偏离程度
        deviation = abs(fair_value - current_price) / current_price
        
        # 基础回归时间 (天)
        base_days = 90  # 3个月基础
        
        # 根据偏离程度调整
        deviation_factor = min(deviation * 2, 1.5)  # 最大增加50%时间
        
        # 根据技术信号调整
        technical_factor = 1.0
        if technical_signals.get('overall_score', 0.5) > 0.7:
            technical_factor = 0.8  # 强势信号加快回归
        elif technical_signals.get('overall_score', 0.5) < 0.3:
            technical_factor = 1.3  # 弱势信号延缓回归
        
        # 根据波动率调整
        volatility_factor = max(0.7, min(1.5, volatility * 5))
        
        estimated_days = int(base_days * deviation_factor * technical_factor * volatility_factor)
        
        return max(7, min(365, estimated_days))  # 限制在7天到1年之间

# 主分析引擎
class StockAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.valuation_engine = ValuationEngine()
        self.technical_engine = TechnicalEngine()
        self.convergence_engine = ConvergenceEngine()
    
    async def analyze_stock(self, symbol: str) -> AnalysisResult:
        """完整股票分析"""
        try:
            async with DataFetcher(self.api_key) as fetcher:
                # 获取数据
                financial_data = await self._fetch_financial_data(fetcher, symbol)
                price_data = await fetcher.get_historical_prices(symbol)
                
                # 估值分析
                fair_value, confidence = self.valuation_engine.dcf_valuation(financial_data)
                
                # 技术分析
                technical_signals = self.technical_engine.analyze_technical_signals(price_data)
                
                # 计算波动率
                volatility = price_data['close'].pct_change().std() * np.sqrt(252)
                
                # 价值回归预测
                convergence_days = self.convergence_engine.estimate_convergence_time(
                    financial_data.current_price, fair_value, volatility, technical_signals
                )
                
                # 判断估值状态
                price_diff = (fair_value - financial_data.current_price) / financial_data.current_price
                if price_diff > 0.15:
                    status = "Significantly Undervalued"
                elif price_diff > 0.05:
                    status = "Undervalued"
                elif price_diff < -0.15:
                    status = "Significantly Overvalued"
                elif price_diff < -0.05:
                    status = "Overvalued"
                else:
                    status = "Fair Valued"
                
                return AnalysisResult(
                    symbol=symbol,
                    current_price=financial_data.current_price,
                    fair_value=fair_value,
                    valuation_status=status,
                    confidence_level=confidence,
                    technical_signals=technical_signals,
                    convergence_estimate=convergence_days
                )
                
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {str(e)}")
            raise StockAnalysisException(f"Failed to analyze {symbol}: {str(e)}")
    
    async def _fetch_financial_data(self, fetcher: DataFetcher, symbol: str) -> FinancialData:
        """获取并处理财务数据"""
        try:
            statements = await fetcher.get_financial_statements(symbol)
            
            # 处理财务数据
            income_data = statements.get('income', [])
            balance_data = statements.get('balance', [])
            cashflow_data = statements.get('cashflow', [])
            profile_data = statements.get('profile', [])
            
            if not income_data or not balance_data or not cashflow_data:
                raise DataFetchError(f"Incomplete financial data for {symbol}")
            
            # 提取关键财务指标
            revenue = [item.get('revenue', 0) for item in income_data[:5]]
            net_income = [item.get('netIncome', 0) for item in income_data[:5]]
            free_cash_flow = [item.get('freeCashFlow', 0) for item in cashflow_data[:5]]
            
            # 最新资产负债表数据
            latest_balance = balance_data[0]
            total_debt = latest_balance.get('totalDebt', 0)
            cash = latest_balance.get('cashAndCashEquivalents', 0)
            shares_outstanding = latest_balance.get('commonStockSharesOutstanding', 1)
            
            # 公司基本信息
            profile = profile_data[0] if profile_data else {}
            beta = profile.get('beta', 1.0)
            current_price = profile.get('price', 0)
            
            return FinancialData(
                symbol=symbol,
                revenue=revenue,
                net_income=net_income,
                free_cash_flow=free_cash_flow,
                total_debt=total_debt,
                cash=cash,
                shares_outstanding=shares_outstanding,
                beta=beta,
                current_price=current_price
            )
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {str(e)}")
            raise DataFetchError(f"Failed to fetch financial data: {str(e)}")

# FastAPI 应用
app = FastAPI(title="股票分析系统", version="1.0.0")

# 全局分析器实例
analyzer = StockAnalyzer(FMP_API_KEY)

@app.get("/")
async def root():
    return {"message": "股票分析系统 API", "version": "1.0.0"}

@app.get("/analysis/{symbol}")
async def get_comprehensive_analysis(symbol: str):
    """获取股票综合分析"""
    try:
        symbol = symbol.upper()
        result = await analyzer.analyze_stock(symbol)
        
        return {
            "symbol": result.symbol,
            "current_price": round(result.current_price, 2),
            "fair_value": round(result.fair_value, 2),
            "valuation_status": result.valuation_status,
            "price_deviation": round((result.fair_value - result.current_price) / result.current_price * 100, 2),
            "confidence_level": round(result.confidence_level * 100, 1),
            "technical_signals": result.technical_signals,
            "estimated_convergence_days": result.convergence_estimate,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/valuation/{symbol}")
async def get_valuation_only(symbol: str):
    """仅获取估值分析"""
    try:
        symbol = symbol.upper()
        async with DataFetcher(FMP_API_KEY) as fetcher:
            financial_data = await analyzer._fetch_financial_data(fetcher, symbol)
            fair_value, confidence = analyzer.valuation_engine.dcf_valuation(financial_data)
            
            price_diff = (fair_value - financial_data.current_price) / financial_data.current_price
            
            return {
                "symbol": symbol,
                "current_price": round(financial_data.current_price, 2),
                "dcf_fair_value": round(fair_value, 2),
                "price_deviation_percent": round(price_diff * 100, 2),
                "confidence_level": round(confidence * 100, 1),
                "recommendation": "BUY" if price_diff > 0.1 else "SELL" if price_diff < -0.1 else "HOLD"
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/technical/{symbol}")
async def get_technical_analysis(symbol: str):
    """获取技术分析"""
    try:
        symbol = symbol.upper()
        async with DataFetcher(FMP_API_KEY) as fetcher:
            price_data = await fetcher.get_historical_prices(symbol)
            
            # 技术指标计算
            support, resistance = analyzer.technical_engine.calculate_support_resistance(price_data)
            ma_data = analyzer.technical_engine.calculate_moving_averages(price_data)
            bb_data = analyzer.technical_engine.calculate_bollinger_bands(price_data)
            signals = analyzer.technical_engine.analyze_technical_signals(price_data)
            
            current_price = price_data['close'].iloc[-1]
            
            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "support_levels": [round(level, 2) for level in support],
                "resistance_levels": [round(level, 2) for level in resistance],
                "moving_averages": {
                    "ma_10": round(ma_data['ma_10'][-1], 2) if ma_data['ma_10'][-1] else None,
                    "ma_20": round(ma_data['ma_20'][-1], 2) if ma_data['ma_20'][-1] else None,
                    "ma_50": round(ma_data['ma_50'][-1], 2) if ma_data['ma_50'][-1] else None
                },
                "bollinger_bands": {
                    "upper": round(bb_data['upper'][-1], 2) if bb_data['upper'][-1] else None,
                    "middle": round(bb_data['middle'][-1], 2) if bb_data['middle'][-1] else None,
                    "lower": round(bb_data['lower'][-1], 2) if bb_data['lower'][-1] else None
                },
                "technical_signals": signals
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/convergence/{symbol}")
async def get_convergence_analysis(symbol: str):
    """获取价值回归分析"""
    try:
        symbol = symbol.upper()
        result = await analyzer.analyze_stock(symbol)
        
        return {
            "symbol": symbol,
            "current_price": round(result.current_price, 2),
            "fair_value": round(result.fair_value, 2),
            "price_gap": round(result.fair_value - result.current_price, 2),
            "estimated_convergence_days": result.convergence_estimate,
            "estimated_convergence_weeks": round(result.convergence_estimate / 7, 1),
            "estimated_convergence_months": round(result.convergence_estimate / 30, 1),
            "confidence_level": round(result.confidence_level * 100, 1)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 批量分析端点
@app.post("/batch_analysis")
async def batch_analysis(symbols: List[str]):
    """批量分析多只股票"""
    if len(symbols) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
    
    results = []
    for symbol in symbols:
        try:
            result = await analyzer.analyze_stock(symbol.upper())
            results.append({
                "symbol": result.symbol,
                "status": "success",
                "current_price": round(result.current_price, 2),
                "fair_value": round(result.fair_value, 2),
                "valuation_status": result.valuation_status,
                "confidence": round(result.confidence_level * 100, 1)
            })
        except Exception as e:
            results.append({
                "symbol": symbol.upper(),
                "status": "error",
                "error": str(e)
            })
    
    return {"results": results}

# 数据可视化生成器
class ChartGenerator:
    @staticmethod
    def create_technical_chart(symbol: str, price_data: pd.DataFrame, 
                             ma_data: Dict, bb_data: Dict, 
                             support: List[float], resistance: List[float]) -> str:
        """生成技术分析图表"""
        
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxis=True,
                           vertical_spacing=0.1,
                           subplot_titles=(f'{symbol} 价格走势', '成交量'),
                           row_width=[0.7, 0.3])
        
        # 价格K线图
        fig.add_trace(go.Candlestick(
            x=price_data['date'],
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name='Price'
        ), row=1, col=1)
        
        # 移动平均线
        if ma_data['ma_10'][-1]:
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=ma_data['ma_10'],
                name='MA10',
                line=dict(color='orange')
            ), row=1, col=1)
        
        if ma_data['ma_20'][-1]:
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=ma_data['ma_20'],
                name='MA20',
                line=dict(color='blue')
            ), row=1, col=1)
        
        # 布林带
        if bb_data['upper'][-1]:
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=bb_data['upper'],
                name='BB Upper',
                line=dict(color='gray', dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=price_data['date'],
                y=bb_data['lower'],
                name='BB Lower',
                line=dict(color='gray', dash='dash'),
                fill='tonexty'
            ), row=1, col=1)
        
        # 支撑和压力位
        for level in support[-3:]:  # 显示最近3个支撑位
            fig.add_hline(y=level, line_dash="dot", line_color="green", 
                         annotation_text=f"Support: ${level:.2f}", row=1, col=1)
        
        for level in resistance[-3:]:  # 显示最近3个压力位
            fig.add_hline(y=level, line_dash="dot", line_color="red", 
                         annotation_text=f"Resistance: ${level:.2f}", row=1, col=1)
        
        # 成交量
        fig.add_trace(go.Bar(
            x=price_data['date'],
            y=price_data['volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} 技术分析图表',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig.to_html()

# 启动脚本
if __name__ == "__main__":
    import uvicorn
    
    # 测试函数
    async def test_analysis():
        """测试分析功能"""
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in test_symbols:
            try:
                print(f"\n正在分析 {symbol}...")
                result = await analyzer.analyze_stock(symbol)
                
                print(f"股票: {result.symbol}")
                print(f"当前价格: ${result.current_price:.2f}")
                print(f"公允价值: ${result.fair_value:.2f}")
                print(f"估值状态: {result.valuation_status}")
                print(f"置信度: {result.confidence_level*100:.1f}%")
                print(f"预计回归时间: {result.convergence_estimate}天")
                print(f"技术信号评分: {result.technical_signals.get('overall_score', 0):.2f}")
                print("-" * 50)
                
            except Exception as e:
                print(f"分析 {symbol} 时出错: {str(e)}")
    
    # 运行测试
    print("开始测试股票分析系统...")
    asyncio.run(test_analysis())
    
    # 启动API服务器
    print("\n启动API服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000)