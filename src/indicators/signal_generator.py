import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .technical_indicators import TechnicalIndicators


class SignalGenerator:
    """Gerador de sinais de compra e venda"""

    def __init__(self, config: Dict):
        self.config = config

    def generate_buy_signal(self, data: pd.DataFrame, index: int) -> Tuple[bool, float, Dict]:
        """Gera sinal de compra para um ponto específico, usando apenas indicadores ativados"""
        # Checagem de índice mínimo para indicadores ativados
        min_index = 0
        if self.config.get('use_ema', True):
            min_index = max(min_index, self.config.get('ema_slow', 15))
        if self.config.get('use_rsi', True):
            min_index = max(min_index, self.config.get('rsi_period', 7))
        if index < min_index + 1:
            return False, 0, {}

        current = data.iloc[index]
        previous = data.iloc[index - 1]

        conditions = {}
        weights = {}
        # EMA
        if self.config.get('use_ema', True):
            conditions['trend_bullish'] = current['ema_fast'] > current['ema_slow']
            weights['trend_bullish'] = 20
            conditions['trend_strong'] = current.get('trend_strength', 0) >= self.config.get('min_trend_strength', 0.005)
            weights['trend_strong'] = 15
        # RSI
        if self.config.get('use_rsi', True):
            conditions['rsi_favorable'] = (
                self.config.get('rsi_oversold', 30) < current['rsi'] < self.config.get('rsi_overbought', 70))
            weights['rsi_favorable'] = 20
            conditions['not_overbought'] = current['rsi'] < self.config.get('rsi_overbought', 70)
            weights['not_overbought'] = 5
        # MACD
        if self.config.get('use_macd', False):
            conditions['macd_bullish'] = current['macd'] > current['macd_signal']
            weights['macd_bullish'] = 15
        # Bollinger
        if self.config.get('use_bollinger', False):
            conditions['price_above_support'] = current['close'] > current['bb_lower']
            weights['price_above_support'] = 10
        # Volume
        if self.config.get('use_volume', True):
            conditions['volume_ok'] = current['volume_ratio'] >= self.config.get('volume_threshold', 1.1)
            weights['volume_ok'] = 10
        # Momentum
        if self.config.get('use_momentum', False):
            conditions['price_momentum'] = current['price_momentum'] > 0
            weights['price_momentum'] = 5

        # Calcular score
        score = sum(weights[condition] for condition, status in conditions.items() if status)
        min_score = self.config.get('min_score', 60)
        should_buy = score >= min_score
        return should_buy, score, conditions

    def generate_sell_signal(self, data: pd.DataFrame, index: int, entry_price: float) -> Tuple[bool, float, Dict, str]:
        """Gera sinal de venda considerando preço de entrada, usando apenas indicadores ativados"""
        current = data.iloc[index]
        previous = data.iloc[index - 1]
        current_price = current['close']
        profit_pct = (current_price - entry_price) / entry_price
        take_profit_pct = self.config.get('take_profit_pct', 0.025)
        stop_loss_pct = self.config.get('stop_loss_pct', 0.015)
        if profit_pct >= take_profit_pct:
            return True, 100, {'take_profit': True}, 'take_profit'
        if profit_pct <= -stop_loss_pct:
            return True, 100, {'stop_loss': True}, 'stop_loss'
        conditions = {}
        weights = {}
        # EMA
        if self.config.get('use_ema', True):
            conditions['trend_bearish'] = current['ema_fast'] < current['ema_slow']
            weights['trend_bearish'] = 35
            conditions['weak_trend'] = current.get('trend_strength', 0) < self.config.get('min_trend_strength', 0.005)
            weights['weak_trend'] = 5
        # RSI
        if self.config.get('use_rsi', True):
            conditions['rsi_overbought'] = current['rsi'] > self.config.get('rsi_overbought', 70)
            weights['rsi_overbought'] = 25
        # MACD
        if self.config.get('use_macd', False):
            conditions['macd_bearish'] = current['macd'] < current['macd_signal']
            weights['macd_bearish'] = 20
        # Bollinger
        if self.config.get('use_bollinger', False):
            conditions['price_near_resistance'] = current['close'] > current['bb_upper'] * 0.99
            weights['price_near_resistance'] = 10
        # Momentum
        if self.config.get('use_momentum', False):
            conditions['price_momentum_down'] = current['price_momentum'] < 0
            weights['price_momentum_down'] = 5
        # Lucro mínimo
        conditions['profitable'] = profit_pct >= self.config.get('min_profit_to_sell', 0.003)
        # Calcular score
        score = sum(weights[condition] for condition, status in conditions.items() if status and condition != 'profitable')
        should_sell = score >= 65 and conditions['profitable']
        reason = 'technical_signal' if should_sell else 'hold'
        return should_sell, score, conditions, reason

    def generate_signals_for_backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gera sinais para todo o dataset (para backtest)"""

        df = data.copy()

        # Inicializar colunas de sinais
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['buy_score'] = 0.0
        df['sell_score'] = 0.0
        df['signal_reason'] = 'none'

        # Simular posição para gerar sinais de venda
        position = False
        entry_price = 0.0
        entry_index = 0

        for i in range(len(df)):
            if not position:
                # Verificar sinal de compra
                should_buy, buy_score, buy_conditions = self.generate_buy_signal(df, i)

                df.iloc[i, df.columns.get_loc('buy_signal')] = should_buy
                df.iloc[i, df.columns.get_loc('buy_score')] = buy_score

                if should_buy:
                    position = True
                    entry_price = df.iloc[i]['close']
                    entry_index = i
                    df.iloc[i, df.columns.get_loc('signal_reason')] = 'buy'

            else:
                # Verificar sinal de venda
                should_sell, sell_score, sell_conditions, reason = self.generate_sell_signal(df, i, entry_price)

                df.iloc[i, df.columns.get_loc('sell_signal')] = should_sell
                df.iloc[i, df.columns.get_loc('sell_score')] = sell_score

                if should_sell:
                    position = False
                    entry_price = 0.0
                    df.iloc[i, df.columns.get_loc('signal_reason')] = reason

        return df