import os
from dataclasses import dataclass
from typing import List, Dict, Any
from binance.client import Client


@dataclass
class TradingConfig:
    """Configurações de trading"""
    operation_code: str = 'ETHBRL'
    asset_code: str = 'ETH'
    candle_interval: str = Client.KLINE_INTERVAL_5MINUTE  # 5 minutos para mais trades
    quantity: float = 0.01  # Ajuste conforme seu capital

    # Take profit e stop loss
    take_profit_percentage: float = 0.005  # 0.5%
    stop_loss_percentage: float = 0.005    # 0.5%
    min_profit_to_sell: float = 0.002      # 0.2%

    # Proteção de capital
    max_daily_loss_percentage: float = 0.10
    max_concurrent_positions: int = 1


@dataclass
class IndicatorConfig:
    """Configurações de indicadores técnicos"""
    fast_ma: int = 5
    slow_ma: int = 12
    rsi_period: int = 7
    volume_ma: int = 20
    bollinger_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Thresholds
    rsi_oversold: int = 20
    rsi_overbought: int = 80
    volume_threshold: float = 1.1
    min_trend_strength: float = 0.2


@dataclass
class BacktestConfig:
    """Configurações de backtest"""
    initial_capital: float = 100.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Período de dados
    days_back: int = 365

    # Validação
    train_ratio: float = 0.7
    validation_ratio: float = 0.2
    test_ratio: float = 0.1

    # Walk-forward
    walk_forward_windows: int = 4

    # Monte Carlo
    monte_carlo_simulations: int = 1000


# Configurações globais
TRADING_CONFIG = TradingConfig()
INDICATOR_CONFIG = IndicatorConfig()
BACKTEST_CONFIG = BacktestConfig()

# API Keys
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(DATA_DIR, "results")


@staticmethod
def get_default_space() -> Dict[str, List[Any]]:
    """Espaço de parâmetros para trades rápidos"""
    return {
        # Médias móveis bem curtas
        'fast_ma_period': [3, 5, 7, 9],
        'slow_ma_period': [10, 12, 15, 18, 21],

        # RSI mais sensível
        'rsi_period': [5, 7, 9],
        'rsi_oversold': [15, 20, 25],
        'rsi_overbought': [75, 80, 85],

        # Stops e targets curtos
        'take_profit_pct': [0.003, 0.005, 0.007, 0.01],   # 0.3% a 1%
        'stop_loss_pct': [0.003, 0.005, 0.007, 0.01],     # 0.3% a 1%

        # Volume e score
        'volume_threshold': [1.0, 1.1, 1.2],
        'min_score': [50, 55, 60]
    }