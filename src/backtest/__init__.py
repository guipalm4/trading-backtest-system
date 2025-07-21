"""
Módulo de backtesting
"""

from .backtest_engine import BacktestEngine
from .trade_manager import TradeManager, Trade
from .performance_analyzer import PerformanceAnalyzer

__all__ = ['BacktestEngine', 'TradeManager', 'Trade', 'PerformanceAnalyzer']