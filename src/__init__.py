"""
Trading Backtest Pro - Sistema Profissional de Backtesting
"""

__version__ = "2.0.0"
__author__ = "Trading Backtest System Team"
# Imports principais para facilitar uso
try:
    from .data.data_manager import DataManager
    from .backtest.backtest_engine import BacktestEngine
    from .optimization.parameter_optimizer import ParameterOptimizer
    from .indicators.technical_indicators import TechnicalIndicators

    __all__ = [
        'DataManager',
        'BacktestEngine',
        'ParameterOptimizer',
        'TechnicalIndicators'
    ]
except ImportError:
    # Se houver problemas de import, não quebrar
    __all__ = []