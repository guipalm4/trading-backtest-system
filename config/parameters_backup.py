from typing import Dict, List, Any


class ParameterSpace:
    """Define o espaço de parâmetros para otimização"""

    @staticmethod
    def get_default_space() -> Dict[str, List[Any]]:
        """Retorna espaço de parâmetros padrão"""
        return {
            # Médias móveis
            'fast_ma_period': list(range(5, 21, 2)),  # [5, 7, 9, 11, 13, 15, 17, 19]
            'slow_ma_period': list(range(15, 51, 5)),  # [15, 20, 25, 30, 35, 40, 45, 50]

            # RSI
            'rsi_period': [10, 12, 14, 16, 18, 20],
            'rsi_oversold': [20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80],

            # Take Profit / Stop Loss
            'take_profit_pct': [0.01, 0.015, 0.02, 0.025, 0.03],
            'stop_loss_pct': [0.005, 0.01, 0.015, 0.02],

            # Volume
            'volume_threshold': [1.0, 1.1, 1.2, 1.3, 1.5],

            # Score mínimo
            'min_score': list(range(40, 81, 5))
        }

    @staticmethod
    def get_conservative_space() -> Dict[str, List[Any]]:
        """Espaço de parâmetros mais conservador"""
        return {
            'fast_ma_period': [9, 12, 15],
            'slow_ma_period': [21, 26, 30],
            'rsi_period': [14, 16, 18],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'take_profit_pct': [0.015, 0.02, 0.025],
            'stop_loss_pct': [0.01, 0.015, 0.02],
            'volume_threshold': [1.1, 1.2, 1.3],
            'min_score': [55, 60, 65, 70]
        }

    @staticmethod
    def get_aggressive_space() -> Dict[str, List[Any]]:
        """Espaço de parâmetros mais agressivo"""
        return {
            'fast_ma_period': [5, 7, 9, 12],
            'slow_ma_period': [15, 18, 21, 26],
            'rsi_period': [10, 12, 14],
            'rsi_oversold': [20, 25, 30],
            'rsi_overbought': [70, 75, 80],
            'take_profit_pct': [0.01, 0.015, 0.02],
            'stop_loss_pct': [0.005, 0.01, 0.015],
            'volume_threshold': [1.0, 1.1, 1.2],
            'min_score': [45, 50, 55, 60]
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