from typing import Dict, List, Any


class ParameterSpace:
    """Define diferentes espaços de parâmetros para cada perfil de estratégia"""

    @staticmethod
    def get_scalping_space() -> Dict[str, List[Any]]:
        """Parâmetros para scalping menos agressivo"""
        return {
            # Médias móveis um pouco mais longas
            'fast_ma_period': [5, 7, 9],
            'slow_ma_period': [12, 15, 18],

            # RSI ainda sensível, mas não extremo
            'rsi_period': [7, 10],
            'rsi_oversold': [20, 25],
            'rsi_overbought': [75, 80],

            # Stops e targets um pouco maiores
            'take_profit_pct': [0.007, 0.01, 0.015],   # 0.7% a 1.5%
            'stop_loss_pct': [0.007, 0.01, 0.015],     # 0.7% a 1.5%

            # Volume mais exigente
            'volume_threshold': [1.2, 1.3, 1.4],

            # Score mais seletivo
            'min_score': [65, 70, 75]
        }

    @staticmethod
    def get_swing_space() -> Dict[str, List[Any]]:
        """Parâmetros para swing trade: operações mais longas, stops e targets maiores"""
        return {
            'fast_ma_period': [9, 12, 14],
            'slow_ma_period': [21, 26, 30],
            'rsi_period': [10, 14],
            'rsi_oversold': [25, 30],
            'rsi_overbought': [70, 75],
            'take_profit_pct': [0.01, 0.015, 0.02],
            'stop_loss_pct': [0.007, 0.01, 0.015],
            'volume_threshold': [1.0, 1.1],
            'min_score': [50, 60]
        }

    @staticmethod
    def get_default_space() -> Dict[str, List[Any]]:
        """Espaço de parâmetros genérico para uso geral ou compatibilidade retroativa"""
        return {
            'fast_ma_period': [5, 7, 9, 12, 14],
            'slow_ma_period': [21, 26, 30, 50],
            'rsi_period': [7, 10, 14],
            'rsi_oversold': [20, 25, 30],
            'rsi_overbought': [70, 75, 80],
            'take_profit_pct': [0.005, 0.01, 0.015, 0.02],
            'stop_loss_pct': [0.005, 0.007, 0.01, 0.015],
            'volume_threshold': [1.0, 1.1, 1.2, 1.3],
            'min_score': [50, 60, 70]
        }


class ParameterValidator:
    """Valida combinações de parâmetros"""

    @staticmethod
    def is_valid_combination(params: Dict[str, Any]) -> bool:
        """Verifica se a combinação de parâmetros é válida"""

        # MA rápida deve ser menor que MA lenta
        if params.get('fast_ma_period', 0) >= params.get('slow_ma_period', 100):
            return False

        # RSI oversold deve ser menor que overbought
        if params.get('rsi_oversold', 0) >= params.get('rsi_overbought', 100):
            return False

        # Take profit deve ser maior que stop loss
        if params.get('take_profit_pct', 1) <= params.get('stop_loss_pct', 0):
            return False

        # Score mínimo deve ser razoável
        min_score = params.get('min_score', 50)
        if min_score < 30 or min_score > 90:
            return False

        return True