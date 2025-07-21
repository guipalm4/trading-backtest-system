import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .technical_indicators import TechnicalIndicators


class SignalGenerator:
    """Gerador de sinais de compra e venda"""

    def __init__(self, config: Dict):
        self.config = config

    def generate_buy_signal(self, data: pd.DataFrame, index: int) -> Tuple[bool, float, Dict]:
        """Gera sinal de compra para um ponto específico"""

        if index < max(self.config.get('slow_ma', 21), self.config.get('rsi_period', 14)) + 1:
            return False, 0, {}

        current = data.iloc[index]
        previous = data.iloc[index - 1]

        # Condições de compra
        conditions = {
            'trend_bullish': current['ema_fast'] > current['ema_slow'],
            'trend_strong': current['trend_strength'] >= self.config.get('min_trend_strength', 0.005),
            'rsi_favorable': (
                        self.config.get('rsi_oversold', 30) < current['rsi'] < self.config.get('rsi_overbought', 70)),
            'macd_bullish': current['macd'] > current['macd_signal'],
            'price_above_support': current['close'] > current['bb_lower'],
            'volume_ok': current['volume_ratio'] >= self.config.get('volume_threshold', 1.1),
            'price_momentum': current['close'] > previous['close'],
            'not_overbought': current['rsi'] < self.config.get('rsi_overbought', 70)
        }

        # Pesos das condições
        weights = {
            'trend_bullish': 20,
            'trend_strong': 15,
            'rsi_favorable': 20,
            'macd_bullish': 15,
            'price_above_support': 10,
            'volume_ok': 10,
            'price_momentum': 5,
            'not_overbought': 5
        }

        # Calcular score
        score = sum(weights[condition] for condition, status in conditions.items() if status)

        # Decisão
        min_score = self.config.get('min_score', 60)
        should_buy = score >= min_score

        return should_buy, score, conditions

    def generate_sell_signal(self, data: pd.DataFrame, index: int, entry_price: float) -> Tuple[bool, float, Dict, str]:
        """Gera sinal de venda considerando preço de entrada"""

        current = data.iloc[index]
        previous = data.iloc[index - 1]

        current_price = current['close']
        profit_pct = (current_price - entry_price) / entry_price

        # Verificar take profit e stop loss primeiro
        take_profit_pct = self.config.get('take_profit_pct', 0.025)
        stop_loss_pct = self.config.get('stop_loss_pct', 0.015)

        if profit_pct >= take_profit_pct:
            return True, 100, {'take_profit': True}, 'take_profit'

        if profit_pct <= -stop_loss_pct:
            return True, 100, {'stop_loss': True}, 'stop_loss'

        # Condições técnicas de venda
        conditions = {
            'trend_bearish': current['ema_fast'] < current['ema_slow'],
            'rsi_overbought': current['rsi'] > self.config.get('rsi_overbought', 70),
            'macd_bearish': current['macd'] < current['macd_signal'],
            'price_near_resistance': current['close'] > current['bb_upper'] * 0.99,
            'weak_trend': current['trend_strength'] < self.config.get('min_trend_strength', 0.005),
            'profitable': profit_pct >= self.config.get('min_profit_to_sell', 0.003),
            'price_momentum_down': current['close'] < previous['close']
        }

        # Pesos das condições
        weights = {
            'trend_bearish': 35,
            'rsi_overbought': 25,
            'macd_bearish': 20,
            'price_near_resistance': 10,
            'weak_trend': 5,
            'price_momentum_down': 5
        }

        # Calcular score
        score = sum(
            weights[condition] for condition, status in conditions.items() if status and condition != 'profitable')

        # Decisão (precisa ser lucrativo E ter sinal técnico)
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