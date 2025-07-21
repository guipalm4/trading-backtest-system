import pytest
import pandas as pd
from src.backtest.backtest_engine import BacktestEngine

@pytest.mark.parametrize("config", [
    # Scalping
    dict(
        use_ema=True, ema_fast=5, ema_slow=12,
        use_rsi=True, rsi_period=7, rsi_oversold=20, rsi_overbought=80,
        use_macd=False, use_bollinger=False, use_volume=True, volume_ma=20, volume_threshold=1.2,
        take_profit_pct=0.007, stop_loss_pct=0.007, min_score=70, quantity=0.01
    ),
    # Swing
    dict(
        use_ema=True, ema_fast=12, ema_slow=26,
        use_rsi=True, rsi_period=14, rsi_oversold=30, rsi_overbought=70,
        use_macd=True, macd_fast=12, macd_slow=26, macd_signal=9,
        use_bollinger=True, bollinger_period=20, use_volume=True, volume_ma=20, volume_threshold=1.1,
        take_profit_pct=0.03, stop_loss_pct=0.015, min_score=60, quantity=0.01
    ),
    # Default
    dict(
        use_ema=True, ema_fast=9, ema_slow=21,
        use_rsi=True, rsi_period=10, rsi_oversold=25, rsi_overbought=75,
        use_macd=False, use_bollinger=True, bollinger_period=20, use_volume=True, volume_ma=20, volume_threshold=1.2,
        take_profit_pct=0.015, stop_loss_pct=0.01, min_score=65, quantity=0.01
    ),
])
def test_backtest_runs_with_various_configs(config, sample_data):
    engine = BacktestEngine(initial_capital=1000)
    results = engine.run_backtest(sample_data, config)
    assert isinstance(results, dict)
    assert 'total_return' in results
    assert 'equity_curve' in results
    assert 'trades_detail' in results 