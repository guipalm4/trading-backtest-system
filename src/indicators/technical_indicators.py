import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging


class TechnicalIndicators:
    """Indicadores técnicos implementados com pandas/numpy (sem TA-Lib)"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)

        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        middle = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(window=period).mean()

        return atr

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    @staticmethod
    def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channels"""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2

        return upper, middle, lower

    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series,
                                period: int = 20) -> pd.Series:
        """Commodity Channel Index (CCI)"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )

        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """Price Momentum"""
        return data.diff(period)

    @staticmethod
    def rate_of_change(data: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change"""
        return ((data - data.shift(period)) / data.shift(period)) * 100

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Calcula todos os indicadores técnicos"""

        result = data.copy()

        try:
            # Médias móveis
            fast_ma = config.get('fast_ma', 9)
            slow_ma = config.get('slow_ma', 21)

            result['ema_fast'] = TechnicalIndicators.ema(data['close'], fast_ma)
            result['ema_slow'] = TechnicalIndicators.ema(data['close'], slow_ma)
            result['sma_volume'] = TechnicalIndicators.sma(data['volume'], 20)

            # RSI
            rsi_period = config.get('rsi_period', 14)
            result['rsi'] = TechnicalIndicators.rsi(data['close'], rsi_period)

            # MACD
            macd_fast = config.get('macd_fast', 12)
            macd_slow = config.get('macd_slow', 26)
            macd_signal = config.get('macd_signal', 9)

            macd_line, signal_line, histogram = TechnicalIndicators.macd(
                data['close'], macd_fast, macd_slow, macd_signal
            )
            result['macd'] = macd_line
            result['macd_signal'] = signal_line
            result['macd_histogram'] = histogram

            # Bollinger Bands
            bb_period = config.get('bb_period', 20)
            bb_std = config.get('bb_std', 2)

            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
                data['close'], bb_period, bb_std
            )
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower

            # ATR
            atr_period = config.get('atr_period', 14)
            result['atr'] = TechnicalIndicators.atr(
                data['high'], data['low'], data['close'], atr_period
            )

            # Stochastic
            stoch_k, stoch_d = TechnicalIndicators.stochastic(
                data['high'], data['low'], data['close']
            )
            result['stoch_k'] = stoch_k
            result['stoch_d'] = stoch_d

            # Williams %R
            result['williams_r'] = TechnicalIndicators.williams_r(
                data['high'], data['low'], data['close']
            )

            # Volume indicators
            if 'volume' in data.columns:
                result['obv'] = TechnicalIndicators.obv(data['close'], data['volume'])
                result['volume_ratio'] = data['volume'] / result['sma_volume']
            else:
                result['obv'] = 0
                result['volume_ratio'] = 1

            # Indicadores customizados
            result['trend_strength'] = (result['ema_fast'] - result['ema_slow']) / result['ema_slow']
            result['price_momentum'] = TechnicalIndicators.momentum(data['close'], 5)
            result['roc'] = TechnicalIndicators.rate_of_change(data['close'], 10)

            # Donchian Channels
            donch_upper, donch_middle, donch_lower = TechnicalIndicators.donchian_channels(
                data['high'], data['low'], 20
            )
            result['donch_upper'] = donch_upper
            result['donch_middle'] = donch_middle
            result['donch_lower'] = donch_lower

            # CCI
            result['cci'] = TechnicalIndicators.commodity_channel_index(
                data['high'], data['low'], data['close']
            )

            logging.info("✅ Todos os indicadores calculados com sucesso")

        except Exception as e:
            logging.error(f"Erro ao calcular indicadores: {e}")
            raise

        return result

    @staticmethod
    def get_available_indicators() -> Dict[str, str]:
        """Retorna lista de indicadores disponíveis"""
        return {
            'sma': 'Simple Moving Average',
            'ema': 'Exponential Moving Average',
            'wma': 'Weighted Moving Average',
            'rsi': 'Relative Strength Index',
            'macd': 'Moving Average Convergence Divergence',
            'bollinger_bands': 'Bollinger Bands',
            'atr': 'Average True Range',
            'stochastic': 'Stochastic Oscillator',
            'williams_r': 'Williams %R',
            'obv': 'On-Balance Volume',
            'donchian_channels': 'Donchian Channels',
            'cci': 'Commodity Channel Index',
            'momentum': 'Price Momentum',
            'roc': 'Rate of Change'
        }