import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_data():
    """Cria dados de exemplo para testes"""

    # Gerar 1000 registros de dados OHLCV
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')

    # Simular preços com random walk
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.01, 1000)
    prices = 50000 * (1 + price_changes).cumprod()

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 1000))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 1000)
    })

    # Garantir que high >= max(open, close) e low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))

    return data


@pytest.fixture
def sample_config():
    """Configuração de exemplo para testes"""
    return {
        'use_ema': True,
        'ema_fast': 7,
        'ema_slow': 15,
        'use_rsi': True,
        'rsi_period': 7,
        'rsi_oversold': 20,
        'rsi_overbought': 75,
        'use_macd': False,
        'use_bollinger': False,
        'use_volume': True,
        'volume_ma': 20,
        'volume_threshold': 1.3,
        'min_trend_strength': 0.2,
        'min_score': 65,
        'take_profit_pct': 0.01,
        'stop_loss_pct': 0.007,
        'min_profit_to_sell': 0.002,
        'quantity': 0.01
    }


@pytest.fixture
def parameter_space():
    """Espaço de parâmetros para testes de otimização"""
    return {
        'fast_ma': [5, 9, 12],
        'slow_ma': [18, 21, 26],
        'take_profit_pct': [0.02, 0.025, 0.03],
        'min_score': [55, 60, 65]
    }