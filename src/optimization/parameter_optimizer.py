import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import random

from ..backtest.backtest_engine import BacktestEngine
from config.parameters import ParameterValidator


class ParameterOptimizer:
    """Otimizador de parâmetros com processamento paralelo"""

    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def generate_parameter_combinations(self, parameter_space: Dict[str, List[Any]],
                                        max_combinations: int = 1000) -> List[Dict[str, Any]]:
        """Gera combinações de parâmetros de forma inteligente"""

        # Calcular total de combinações possíveis
        total_combinations = 1
        for values in parameter_space.values():
            total_combinations *= len(values)

        logging.info(f"Total de combinações possíveis: {total_combinations:,}")

        if total_combinations <= max_combinations:
            # Se não há muitas combinações, usar todas
            keys = list(parameter_space.keys())
            values = list(parameter_space.values())
            all_combinations = list(product(*values))

            parameter_sets = []
            for combo in all_combinations:
                param_dict = dict(zip(keys, combo))
                if ParameterValidator.is_valid_combination(param_dict):
                    parameter_sets.append(param_dict)

            logging.info(f"Combinações válidas: {len(parameter_sets)}")
            return parameter_sets

        else:
            # Usar amostragem inteligente
            return self._intelligent_sampling(parameter_space, max_combinations)

    def _intelligent_sampling(self, parameter_space: Dict[str, List[Any]],
                              max_combinations: int) -> List[Dict[str, Any]]:
        """Amostragem inteligente de parâmetros"""

        parameter_sets = []
        keys = list(parameter_space.keys())

        # 1. Amostragem aleatória (60%)
        random_count = int(max_combinations * 0.6)
        for _ in range(random_count * 2):  # Gerar mais para filtrar inválidos
            param_dict = {}
            for key, values in parameter_space.items():
                param_dict[key] = random.choice(values)

            if ParameterValidator.is_valid_combination(param_dict):
                parameter_sets.append(param_dict)
                if len(parameter_sets) >= random_count:
                    break

        # 2. Amostragem em grid (25%)
        grid_count = int(max_combinations * 0.25)
        grid_combinations = self._generate_grid_sample(parameter_space, grid_count)
        parameter_sets.extend(grid_combinations)

        # 3. Configurações baseadas em conhecimento (15%)
        expert_count = int(max_combinations * 0.15)
        expert_combinations = self._generate_expert_combinations(parameter_space, expert_count)
        parameter_sets.extend(expert_combinations)

        # Remover duplicatas
        unique_sets = []
        seen = set()
        for params in parameter_sets:
            param_tuple = tuple(sorted(params.items()))
            if param_tuple not in seen:
                seen.add(param_tuple)
                unique_sets.append(params)

        # Limitar ao máximo solicitado
        final_sets = unique_sets[:max_combinations]

        logging.info(f"Combinações geradas: {len(final_sets)}")
        return final_sets

    def _generate_grid_sample(self, parameter_space: Dict[str, List[Any]],
                              count: int) -> List[Dict[str, Any]]:
        """Gera amostra em grid (valores extremos e médios)"""

        grid_sets = []

        for key, values in parameter_space.items():
            if len(values) >= 3:
                # Pegar início, meio e fim
                grid_values = [values[0], values[len(values) // 2], values[-1]]
            else:
                grid_values = values

            # Criar combinações com outros parâmetros no valor médio
            base_params = {}
            for other_key, other_values in parameter_space.items():
                if other_key != key:
                    base_params[other_key] = other_values[len(other_values) // 2]

            for value in grid_values:
                params = base_params.copy()
                params[key] = value

                if ParameterValidator.is_valid_combination(params):
                    grid_sets.append(params)
                    if len(grid_sets) >= count:
                        break

            if len(grid_sets) >= count:
                break

        return grid_sets[:count]

    def _generate_expert_combinations(self, parameter_space: Dict[str, List[Any]],
                                      count: int) -> List[Dict[str, Any]]:
        """Gera combinações baseadas em conhecimento de trading"""

        expert_sets = []

        # Configurações conhecidas que funcionam bem
        expert_configs = [
            # Configuração conservadora
            {
                'fast_ma_period': 9,
                'slow_ma_period': 21,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'take_profit_pct': 0.02,
                'stop_loss_pct': 0.01,
                'volume_threshold': 1.2,
                'min_score': 65
            },
            # Configuração agressiva
            {
                'fast_ma_period': 5,
                'slow_ma_period': 15,
                'rsi_period': 10,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'take_profit_pct': 0.015,
                'stop_loss_pct': 0.005,
                'volume_threshold': 1.1,
                'min_score': 55
            },
            # Configuração balanceada
            {
                'fast_ma_period': 12,
                'slow_ma_period': 26,
                'rsi_period': 16,
                'rsi_oversold': 28,
                'rsi_overbought': 72,
                'take_profit_pct': 0.025,
                'stop_loss_pct': 0.015,
                'volume_threshold': 1.3,
                'min_score': 60
            }
        ]

        # Filtrar apenas parâmetros que existem no espaço
        for config in expert_configs:
            filtered_config = {}
            for key, value in config.items():
                if key in parameter_space and value in parameter_space[key]:
                    filtered_config[key] = value

            if len(filtered_config) >= len(parameter_space) * 0.7:  # Pelo menos 70% dos parâmetros
                # Completar parâmetros faltantes com valores médios
                for key in parameter_space:
                    if key not in filtered_config:
                        values = parameter_space[key]
                        filtered_config[key] = values[len(values) // 2]

                if ParameterValidator.is_valid_combination(filtered_config):
                    expert_sets.append(filtered_config)

        # Gerar variações das configurações expert
        while len(expert_sets) < count and expert_sets:
            base_config = random.choice(expert_sets[:3])  # Usar apenas as 3 primeiras
            variation = base_config.copy()

            # Variar 1-2 parâmetros aleatoriamente
            params_to_vary = random.sample(list(parameter_space.keys()),
                                           random.randint(1, min(2, len(parameter_space))))

            for param in params_to_vary:
                variation[param] = random.choice(parameter_space[param])

            if ParameterValidator.is_valid_combination(variation):
                expert_sets.append(variation)

        return expert_sets[:count]

    def _run_single_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Executa um único backtest"""

        try:
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission=self.commission
            )

            results = engine.run_backtest(data, params)

            # Adicionar parâmetros ao resultado
            result_with_params = results.copy()
            for key, value in params.items():
                result_with_params[f'param_{key}'] = value

            return result_with_params

        except Exception as e:
            logging.error(f"Erro no backtest: {e}")
            return {
                'total_return': -1,
                'max_drawdown': -1,
                'sharpe_ratio': -10,
                'total_trades': 0,
                'error': str(e)
            }

    def optimize_parameters(self, data: pd.DataFrame, parameter_space: Dict[str, List[Any]],
                            max_combinations: int = 500, n_jobs: int = -1, score_weights: dict = None) -> Tuple[
        Dict[str, Any], List[Dict[str, Any]]]:
        """Executa otimização completa de parâmetros"""

        logging.info("🔬 Iniciando otimização de parâmetros")

        # Gerar combinações
        parameter_combinations = self.generate_parameter_combinations(parameter_space, max_combinations)

        logging.info(f"📊 Testando {len(parameter_combinations)} combinações")

        # Executar backtests em paralelo
        if n_jobs == -1:
            n_jobs = min(4, len(parameter_combinations))  # Limitar para não sobrecarregar

        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(self._run_single_backtest)(data, params)
            for params in tqdm(parameter_combinations, desc="Executando backtests")
        )

        # Filtrar resultados válidos
        valid_results = [r for r in results if r.get('total_trades', 0) > 0 and 'error' not in r]

        if not valid_results:
            raise ValueError("Nenhum backtest válido foi executado")

        logging.info(f"✅ Backtests válidos: {len(valid_results)}/{len(results)}")

        # Encontrar melhor configuração
        best_result = self._select_best_configuration(valid_results, score_weights=score_weights)

        # Extrair parâmetros da melhor configuração
        best_params = {}
        for key, value in best_result.items():
            if key.startswith('param_'):
                param_name = key.replace('param_', '')
                best_params[param_name] = value

        logging.info("🏆 Melhor configuração encontrada:")
        for param, value in best_params.items():
            logging.info(f"   {param}: {value}")

        logging.info(f"📈 Performance: {best_result['total_return']:.2%} retorno, "
                     f"{best_result['max_drawdown']:.2%} drawdown, "
                     f"{best_result['sharpe_ratio']:.3f} Sharpe")

        return best_params, valid_results

    def _select_best_configuration(self, results: List[Dict[str, Any]], score_weights: dict = None) -> Dict[str, Any]:
        """Seleciona a melhor configuração usando múltiplos critérios e pesos configuráveis"""

        df = pd.DataFrame(results)

        # Filtros de qualidade mínima
        quality_filter = (
                (df['total_trades'] >= 10) &  # Mínimo de trades
                (df['max_drawdown'] >= -0.4) &  # Drawdown máximo 40%
                (df['total_return'] > -0.2)  # Perda máxima 20%
        )

        df_filtered = df[quality_filter].copy()

        if df_filtered.empty:
            logging.warning("⚠️ Nenhuma configuração passou nos filtros de qualidade")
            # Usar todas as configurações, mas com filtros mais relaxados
            df_filtered = df[df['total_trades'] >= 5].copy()

        if df_filtered.empty:
            # Última tentativa - usar a melhor por retorno
            return df.loc[df['total_return'].idxmax()].to_dict()

        # Normalizar métricas para score composto
        metrics_to_normalize = ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor']

        for metric in metrics_to_normalize:
            if metric in df_filtered.columns:
                min_val = df_filtered[metric].min()
                max_val = df_filtered[metric].max()
                if max_val > min_val:
                    df_filtered[f'norm_{metric}'] = (df_filtered[metric] - min_val) / (max_val - min_val)
                else:
                    df_filtered[f'norm_{metric}'] = 0.5

        # Normalizar drawdown (inverso - menor é melhor)
        min_dd = df_filtered['max_drawdown'].min()
        max_dd = df_filtered['max_drawdown'].max()
        if max_dd > min_dd:
            df_filtered['norm_max_drawdown'] = 1 - ((df_filtered['max_drawdown'] - min_dd) / (max_dd - min_dd))
        else:
            df_filtered['norm_max_drawdown'] = 0.5

        # Pesos configuráveis por perfil
        if score_weights is None:
            score_weights = {
                'total_return': 0.30,
                'max_drawdown': 0.25,
                'sharpe_ratio': 0.20,
                'win_rate': 0.15,
                'profit_factor': 0.10
            }

        df_filtered['composite_score'] = (
            df_filtered.get('norm_total_return', 0) * score_weights.get('total_return', 0) +
            df_filtered.get('norm_max_drawdown', 0) * score_weights.get('max_drawdown', 0) +
            df_filtered.get('norm_sharpe_ratio', 0) * score_weights.get('sharpe_ratio', 0) +
            df_filtered.get('norm_win_rate', 0) * score_weights.get('win_rate', 0) +
            df_filtered.get('norm_profit_factor', 0) * score_weights.get('profit_factor', 0)
        )

        # Retornar melhor configuração
        best_idx = df_filtered['composite_score'].idxmax()
        return df_filtered.loc[best_idx].to_dict()

    def analyze_parameter_sensitivity(self, results: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Analisa sensibilidade dos parâmetros"""

        df = pd.DataFrame(results)

        # Identificar colunas de parâmetros
        param_columns = [col for col in df.columns if col.startswith('param_')]

        sensitivity_analysis = {}

        for param_col in param_columns:
            param_name = param_col.replace('param_', '')

            # Agrupar por valor do parâmetro
            grouped = df.groupby(param_col).agg({
                'total_return': ['mean', 'std', 'count'],
                'max_drawdown': 'mean',
                'sharpe_ratio': 'mean',
                'win_rate': 'mean'
            }).round(4)

            # Calcular correlação
            correlation = df[param_col].corr(df['total_return'])

            sensitivity_analysis[param_name] = {
                'correlation_with_return': correlation,
                'statistics_by_value': grouped.to_dict(),
                'best_value': df.loc[df['total_return'].idxmax(), param_col],
                'worst_value': df.loc[df['total_return'].idxmin(), param_col]
            }

        return sensitivity_analysis