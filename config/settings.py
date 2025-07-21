import os
from dataclasses import dataclass
from typing import Optional
from binance.client import Client


@dataclass
class StrategyConfig:
    # Trading
    operation_code: str = 'ETHBRL'
    asset_code: str = 'ETH'
    candle_interval: str = Client.KLINE_INTERVAL_5MINUTE
    quantity: float = 0.01
    take_profit_pct: float = 0.01
    stop_loss_pct: float = 0.007
    min_profit_to_sell: float = 0.002
    max_daily_loss_percentage: float = 0.10
    max_concurrent_positions: int = 1

    # Indicadores - flags de ativação e parâmetros
    use_ema: bool = True
    ema_fast: int = 7
    ema_slow: int = 15
    use_rsi: bool = True
    rsi_period: int = 7
    rsi_oversold: int = 20
    rsi_overbought: int = 75
    use_macd: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_bollinger: bool = False
    bollinger_period: int = 20
    use_volume: bool = True
    volume_ma: int = 20
    volume_threshold: float = 1.3
    use_trend: bool = True
    min_trend_strength: float = 0.2

    # Sinais
    min_score: int = 65

    # Outros parâmetros podem ser adicionados conforme necessário

# Instância global de configuração
STRATEGY_CONFIG = StrategyConfig()

# API Keys
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(DATA_DIR, "results")