import os
from dataclasses import dataclass
from typing import List, Dict
from binance.client import Client


@dataclass
class TradingConfig:
    """Configurações de trading"""
    operation_code: str = 'ETHBRL'
    asset_code: str = 'ETH'
    candle_interval: str = Client.KLINE_INTERVAL_15MINUTE
    quantity: float = 0.0003

    # Take profit e stop loss
    take_profit_percentage: float = 0.025  # 2.5%
    stop_loss_percentage: float = 0.015  # 1.5%
    min_profit_to_sell: float = 0.003  # 0.3%

    # Proteção de capital
    max_daily_loss_percentage: float = 0.10  # 10%
    max_concurrent_positions: int = 1


@dataclass
class IndicatorConfig:
    """Configurações de indicadores técnicos"""
    fast_ma: int = 9
    slow_ma: int = 21
    rsi_period: int = 14
    volume_ma: int = 20
    bollinger_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Thresholds
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    volume_threshold: float = 1.1
    min_trend_strength: float = 0.005


@dataclass
class BacktestConfig:
    """Configurações de backtest"""
    initial_capital: float = 93.0
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