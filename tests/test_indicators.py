import pytest
import pandas as pd
import numpy as np

from src.indicators.technical_indicators import TechnicalIndicators
from src.indicators.signal_generator import SignalGenerator


class TestTechnicalIndicators:
    """Testes para indicadores técnicos"""

    def test_ema(self, sample_data):
        """Testa cálculo de EMA"""
        ema = TechnicalIndicators.ema(sample_data['close'], 10)

        assert len(ema) == len(sample_data)
        assert not ema.isna().all()
        assert ema.iloc[-1] != ema.iloc[0]  # Deve variar

    def test_sma(self, sample_data):
        """Testa cálculo de SMA"""
        sma = TechnicalIndicators.sma(sample_data['close'], 10)

        assert len(sma) == len(sample_data)
        assert sma.iloc[:9].isna().all()  # Primeiros 9 devem ser NaN
        assert not sma.iloc[10:].isna().any()  # Resto não deve ser NaN

    def test_rsi(self, sample_data):
        """Testa cálculo de RSI"""
        rsi = TechnicalIndicators.rsi(sample_data['close'], 14)

        assert len(rsi) == len(sample_data)
        assert (rsi >= 0).all() or rsi.isna().all()
        assert (rsi <= 100).all() or rsi.isna().all()

    def test_macd(self, sample_data):
        """Testa cálculo de MACD"""
        macd_line, signal_line, histogram = TechnicalIndicators.macd(sample_data['close'])

        assert len(macd_line) == len(sample_data)
        assert len(signal_line) == len(sample_data)
        assert len(histogram) == len(sample_data)

        # Verificar relação: histogram = macd - signal
        valid_idx = ~(macd_line.isna() | signal_line.isna() | histogram.isna())
        if valid_idx.any():
            np.testing.assert_array_almost_equal(
                histogram[valid_idx],
                macd_line[valid_idx] - signal_line[valid_idx],
                decimal=10
            )

    def test_bollinger_bands(self, sample_data):
        """Testa cálculo de Bollinger Bands"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(sample_data['close'])

        assert len(upper) == len(sample_data)
        assert len(middle) == len(sample_data)
        assert len(lower) == len(sample_data)

        # Verificar ordem: upper >= middle >= lower
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_idx.any():
            assert (upper[valid_idx] >= middle[valid_idx]).all()
            assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_atr(self, sample_data):
        """Testa cálculo de ATR"""
        atr = TechnicalIndicators.atr(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        assert len(atr) == len(sample_data)
        assert (atr >= 0).all() or atr.isna().all()

    def test_stochastic(self, sample_data):
        """Testa cálculo de Stochastic"""
        k_percent, d_percent = TechnicalIndicators.stochastic(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        assert len(k_percent) == len(sample_data)
        assert len(d_percent) == len(sample_data)

        # Verificar range 0-100
        valid_k = ~k_percent.isna()
        valid_d = ~d_percent.isna()

        if valid_k.any():
            assert (k_percent[valid_k] >= 0).all()
            assert (k_percent[valid_k] <= 100).all()

        if valid_d.any():
            assert (d_percent[valid_d] >= 0).all()
            assert (d_percent[valid_d] <= 100).all()

    def test_williams_r(self, sample_data):
        """Testa cálculo de Williams %R"""
        wr = TechnicalIndicators.williams_r(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        assert len(wr) == len(sample_data)

        # Verificar range -100 a 0
        valid_idx = ~wr.isna()
        if valid_idx.any():
            assert (wr[valid_idx] >= -100).all()
            assert (wr[valid_idx] <= 0).all()

    def test_calculate_all_indicators(self, sample_data, sample_config):
        """Testa cálculo de todos os indicadores"""
        result = TechnicalIndicators.calculate_all_indicators(sample_data, sample_config)

        # Verificar se todas as colunas esperadas existem
        expected_columns = [
            'ema_fast', 'ema_slow', 'sma_volume', 'rsi',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'volume_ratio', 'trend_strength', 'price_momentum'
        ]

        for col in expected_columns:
            assert col in result.columns

        assert len(result) == len(sample_data)


class TestSignalGenerator:
    """Testes para gerador de sinais"""

    def setup_method(self):
        """Setup para cada teste"""
        self.config = {
            'fast_ma': 9,
            'slow_ma': 21,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'take_profit_pct': 0.025,
            'stop_loss_pct': 0.015,
            'volume_threshold': 1.2,
            'min_score': 60,
            'min_trend_strength': 0.005,
            'min_profit_to_sell': 0.003
        }
        self.signal_generator = SignalGenerator(self.config)

    def test_generate_buy_signal(self, sample_data):
        """Testa geração de sinal de compra"""
        # Adicionar indicadores necessários
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(
            sample_data, self.config
        )

        # Testar em índice válido
        index = 50
        should_buy, score, conditions = self.signal_generator.generate_buy_signal(
            data_with_indicators, index
        )

        assert isinstance(should_buy, bool)
        assert isinstance(score, (int, float))
        assert isinstance(conditions, dict)
        assert 0 <= score <= 100

    def test_generate_sell_signal(self, sample_data):
        """Testa geração de sinal de venda"""
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(
            sample_data, self.config
        )

        index = 50
        entry_price = data_with_indicators.iloc[index]['close']

        should_sell, score, conditions, reason = self.signal_generator.generate_sell_signal(
            data_with_indicators, index, entry_price
        )

        assert isinstance(should_sell, bool)
        assert isinstance(score, (int, float))
        assert isinstance(conditions, dict)
        assert isinstance(reason, str)
        assert reason in ['take_profit', 'stop_loss', 'technical_signal', 'hold']

    def test_take_profit_trigger(self, sample_data):
        """Testa trigger de take profit"""
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(
            sample_data, self.config
        )

        index = 50
        current_price = data_with_indicators.iloc[index]['close']

        # Simular entrada com preço menor para garantir take profit
        entry_price = current_price * 0.95  # 5% abaixo do preço atual

        should_sell, score, conditions, reason = self.signal_generator.generate_sell_signal(
            data_with_indicators, index, entry_price
        )

        # Se o lucro for >= take_profit_pct, deve vender
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct >= self.config['take_profit_pct']:
            assert should_sell
            assert reason == 'take_profit'

    def test_stop_loss_trigger(self, sample_data):
        """Testa trigger de stop loss"""
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(
            sample_data, self.config
        )

        index = 50
        current_price = data_with_indicators.iloc[index]['close']

        # Simular entrada com preço maior para garantir stop loss
        entry_price = current_price * 1.05  # 5% acima do preço atual

        should_sell, score, conditions, reason = self.signal_generator.generate_sell_signal(
            data_with_indicators, index, entry_price
        )

        # Se a perda for >= stop_loss_pct, deve vender
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct <= -self.config['stop_loss_pct']:
            assert should_sell
            assert reason == 'stop_loss'

    def test_generate_signals_for_backtest(self, sample_data):
        """Testa geração de sinais para backtest completo"""
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(
            sample_data, self.config
        )

        result = self.signal_generator.generate_signals_for_backtest(data_with_indicators)

        # Verificar se colunas de sinais foram adicionadas
        expected_columns = [
            'buy_signal', 'sell_signal', 'buy_score',
            'sell_score', 'signal_reason'
        ]

        for col in expected_columns:
            assert col in result.columns

        assert len(result) == len(data_with_indicators)

        # Verificar tipos de dados
        assert result['buy_signal'].dtype == bool
        assert result['sell_signal'].dtype == bool
        assert pd.api.types.is_numeric_dtype(result['buy_score'])
        assert pd.api.types.is_numeric_dtype(result['sell_score'])

    def test_signal_consistency(self, sample_data):
        """Testa consistência dos sinais"""
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(
            sample_data, self.config
        )

        result = self.signal_generator.generate_signals_for_backtest(data_with_indicators)

        # Não deve ter buy e sell signal no mesmo momento
        simultaneous_signals = (result['buy_signal'] & result['sell_signal']).sum()
        assert simultaneous_signals == 0

        # Scores devem estar no range válido
        assert (result['buy_score'] >= 0).all()
        assert (result['buy_score'] <= 100).all()
        assert (result['sell_score'] >= 0).all()
        assert (result['sell_score'] <= 100).all()