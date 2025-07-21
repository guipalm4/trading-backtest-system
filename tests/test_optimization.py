import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.optimization.parameter_optimizer import ParameterOptimizer
from src.optimization.walk_forward import WalkForwardAnalyzer
from src.optimization.monte_carlo import MonteCarloValidator


class TestParameterOptimizer:
    """Testes para otimização de parâmetros"""

    def setup_method(self):
        """Setup para cada teste"""
        self.optimizer = ParameterOptimizer(initial_capital=10000)

    def test_init(self):
        """Testa inicialização do otimizador"""
        assert self.optimizer.initial_capital == 10000
        assert self.optimizer.commission == 0.001

    def test_generate_parameter_combinations_small(self, parameter_space):
        """Testa geração de combinações pequenas"""
        combinations = self.optimizer.generate_parameter_combinations(
            parameter_space, max_combinations=100
        )

        assert len(combinations) > 0
        assert len(combinations) <= 100

        # Verificar estrutura das combinações
        for combo in combinations:
            assert isinstance(combo, dict)
            for key in parameter_space.keys():
                assert key in combo
                assert combo[key] in parameter_space[key]

    def test_generate_parameter_combinations_large(self):
        """Testa geração com espaço grande de parâmetros"""
        large_space = {
            'param1': list(range(1, 21)),  # 20 valores
            'param2': list(range(1, 21)),  # 20 valores
            'param3': list(range(1, 21))  # 20 valores = 8000 combinações
        }

        combinations = self.optimizer.generate_parameter_combinations(
            large_space, max_combinations=100
        )

        assert len(combinations) == 100  # Deve limitar a 100

    @patch('optimization.parameter_optimizer.BacktestEngine')
    def test_run_single_backtest(self, mock_engine_class, sample_data, sample_config):
        """Testa execução de backtest único"""
        # Mock do engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run_backtest.return_value = {
            'total_return': 0.15,
            'max_drawdown': -0.08,
            'sharpe_ratio': 1.2,
            'total_trades': 10
        }

        result = self.optimizer._run_single_backtest(sample_data, sample_config)

        assert 'total_return' in result
        assert 'param_fast_ma' in result  # Parâmetros devem ser adicionados
        mock_engine.run_backtest.assert_called_once()

    @patch('optimization.parameter_optimizer.ParameterOptimizer._run_single_backtest')
    def test_optimize_parameters(self, mock_backtest, sample_data, parameter_space):
        """Testa otimização completa"""
        # Mock dos resultados
        mock_results = []
        for i in range(10):
            mock_results.append({
                'total_return': np.random.uniform(-0.1, 0.3),
                'max_drawdown': np.random.uniform(-0.3, -0.05),
                'sharpe_ratio': np.random.uniform(-1, 2),
                'total_trades': np.random.randint(5, 50),
                'param_fast_ma': 9,
                'param_slow_ma': 21
            })

        mock_backtest.side_effect = mock_results

        best_params, results = self.optimizer.optimize_parameters(
            sample_data, parameter_space, max_combinations=10, n_jobs=1
        )

        assert isinstance(best_params, dict)
        assert isinstance(results, list)
        assert len(results) <= 10
        assert 'fast_ma' in best_params
        assert 'slow_ma' in best_params

    def test_select_best_configuration(self):
        """Testa seleção da melhor configuração"""
        results = [
            {
                'total_return': 0.15, 'max_drawdown': -0.08, 'sharpe_ratio': 1.2,
                'total_trades': 20, 'win_rate': 0.6, 'profit_factor': 1.5
            },
            {
                'total_return': 0.25, 'max_drawdown': -0.25, 'sharpe_ratio': 0.8,
                'total_trades': 15, 'win_rate': 0.5, 'profit_factor': 1.8
            },
            {
                'total_return': 0.10, 'max_drawdown': -0.05, 'sharpe_ratio': 1.8,
                'total_trades': 25, 'win_rate': 0.7, 'profit_factor': 1.3
            }
        ]

        best = self.optimizer._select_best_configuration(results)

        assert isinstance(best, dict)
        assert 'total_return' in best
        # Deve balancear retorno, risco e qualidade


class TestWalkForwardAnalyzer:
    """Testes para análise walk-forward"""

    def setup_method(self):
        """Setup para cada teste"""
        self.analyzer = WalkForwardAnalyzer()

    def test_split_data_for_walk_forward(self, sample_data):
        """Testa divisão de dados para walk-forward"""
        window_pairs = self.analyzer.split_data_for_walk_forward(
            sample_data, n_windows=3, optimization_length_pct=0.7
        )

        assert len(window_pairs) > 0
        assert len(window_pairs) <= 3

        for train_data, test_data in window_pairs:
            assert isinstance(train_data, pd.DataFrame)
            assert isinstance(test_data, pd.DataFrame)
            assert len(train_data) > 0
            assert len(test_data) > 0
            # Dados de teste devem vir depois dos de treino
            assert train_data['timestamp'].max() <= test_data['timestamp'].min()

    @patch('optimization.walk_forward.ParameterOptimizer')
    @patch('optimization.walk_forward.BacktestEngine')
    def test_run_walk_forward_analysis(self, mock_engine_class, mock_optimizer_class,
                                       sample_data, parameter_space):
        """Testa análise walk-forward completa"""
        # Mock do otimizador
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize_parameters.return_value = (
            {'fast_ma': 9, 'slow_ma': 21},  # best_params
            [{'total_return': 0.15}]  # optimization_results
        )

        # Mock do engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        mock_engine.run_backtest.return_value = {
            'total_return': 0.12,
            'max_drawdown': -0.08,
            'sharpe_ratio': 1.1,
            'total_trades': 8,
            'win_rate': 0.625,
            'profit_factor': 1.4,
            'expectancy': 50
        }

        results = self.analyzer.run_walk_forward_analysis(
            sample_data, parameter_space, n_windows=2, max_combinations_per_window=10
        )

        assert isinstance(results, list)
        assert len(results) > 0

        for result in results:
            assert 'window' in result
            assert 'out_sample_return' in result
            assert 'return_degradation' in result

    def test_analyze_consistency(self):
        """Testa análise de consistência"""
        wf_results = [
            {
                'window': 1, 'out_sample_return': 0.15, 'out_sample_drawdown': -0.08,
                'out_sample_sharpe': 1.2, 'return_degradation': -0.02
            },
            {
                'window': 2, 'out_sample_return': 0.10, 'out_sample_drawdown': -0.12,
                'out_sample_sharpe': 0.9, 'return_degradation': -0.05
            },
            {
                'window': 3, 'out_sample_return': 0.08, 'out_sample_drawdown': -0.06,
                'out_sample_sharpe': 1.1, 'return_degradation': -0.03
            }
        ]

        consistency = self.analyzer.analyze_consistency(wf_results)

        assert 'avg_return' in consistency
        assert 'positive_periods' in consistency
        assert 'consistency_ratio' in consistency
        assert 'overall_grade' in consistency

        assert consistency['total_periods'] == 3
        assert consistency['positive_periods'] == 3  # Todos positivos
        assert consistency['consistency_ratio'] == 1.0


class TestMonteCarloValidator:
    """Testes para validação Monte Carlo"""

    def setup_method(self):
        """Setup para cada teste"""
        self.validator = MonteCarloValidator()

    def test_bootstrap_data(self, sample_data):
        """Testa amostragem bootstrap"""
        bootstrapped = self.validator.bootstrap_data(sample_data, sample_ratio=0.8)

        assert isinstance(bootstrapped, pd.DataFrame)
        assert len(bootstrapped) == int(len(sample_data) * 0.8)
        assert list(bootstrapped.columns) == list(sample_data.columns)
        # Dados devem estar ordenados por timestamp
        assert bootstrapped['timestamp'].is_monotonic_increasing

    def test_block_bootstrap_data(self, sample_data):
        """Testa amostragem block bootstrap"""
        block_bootstrapped = self.validator.block_bootstrap_data(
            sample_data, block_size=20, sample_ratio=0.8
        )

        assert isinstance(block_bootstrapped, pd.DataFrame)
        assert len(block_bootstrapped) <= int(len(sample_data) * 0.8)
        assert list(block_bootstrapped.columns) == list(sample_data.columns)

    def test_noise_injection_data(self, sample_data):
        """Testa injeção de ruído"""
        noisy_data = self.validator.noise_injection_data(sample_data, noise_level=0.01)

        assert isinstance(noisy_data, pd.DataFrame)
        assert len(noisy_data) == len(sample_data)
        assert list(noisy_data.columns) == list(sample_data.columns)

        # Dados com ruído devem ser diferentes dos originais
        assert not noisy_data['close'].equals(sample_data['close'])
        # Mas não muito diferentes
        price_diff = abs(noisy_data['close'] - sample_data['close']) / sample_data['close']
        assert price_diff.mean() < 0.02  # Menos de 2% de diferença média

    @patch('optimization.monte_carlo.BacktestEngine')
    def test_run_monte_carlo_simulation(self, mock_engine_class, sample_data, sample_config):
        """Testa simulação Monte Carlo"""
        # Mock do engine
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine

        # Simular resultados variados
        def mock_backtest(data, config):
            return {
                'total_return': np.random.normal(0.1, 0.05),
                'max_drawdown': np.random.uniform(-0.2, -0.05),
                'sharpe_ratio': np.random.normal(1.0, 0.3),
                'total_trades': np.random.randint(5, 25),
                'win_rate': np.random.uniform(0.4, 0.7),
                'profit_factor': np.random.uniform(1.0, 2.5),
                'expectancy': np.random.normal(20, 10),
                'net_profit': np.random.normal(1000, 500),
                'largest_win': np.random.uniform(100, 500),
                'largest_loss': np.random.uniform(-300, -50),
                'avg_trade_duration': pd.Timedelta(hours=np.random.uniform(1, 6))
            }

        mock_engine.run_backtest.side_effect = mock_backtest

        results = self.validator.run_monte_carlo_simulation(
            sample_data, sample_config, n_simulations=10
        )

        assert isinstance(results, list)
        assert len(results) <= 10  # Pode ter menos se houver falhas

        for result in results:
            assert 'simulation' in result
            assert 'total_return' in result
            assert 'max_drawdown' in result
            assert 'sharpe_ratio' in result

    def test_calculate_statistics(self):
        """Testa cálculo de estatísticas Monte Carlo"""
        mc_results = []
        np.random.seed(42)

        for i in range(100):
            mc_results.append({
                'simulation': i + 1,
                'total_return': np.random.normal(0.1, 0.05),
                'max_drawdown': np.random.uniform(-0.2, -0.05),
                'sharpe_ratio': np.random.normal(1.0, 0.3),
                'total_trades': np.random.randint(5, 25),
                'win_rate': np.random.uniform(0.4, 0.7),
                'profit_factor': np.random.uniform(1.0, 2.5)
            })

        statistics = self.validator.calculate_statistics(mc_results)

        # Verificar estatísticas básicas
        assert 'mean_return' in statistics
        assert 'std_return' in statistics
        assert 'var_95' in statistics
        assert 'cvar_95' in statistics
        assert 'probability_profit' in statistics

        # Verificar valores razoáveis
        assert 0 <= statistics['probability_profit'] <= 1
        assert statistics['var_95'] <= statistics['percentile_5']
        assert statistics['mean_return'] > statistics['var_95']

    def test_confidence_intervals(self):
        """Testa cálculo de intervalos de confiança"""
        mc_results = []
        np.random.seed(42)

        for i in range(50):
            mc_results.append({
                'total_return': np.random.normal(0.1, 0.02),
                'max_drawdown': np.random.uniform(-0.15, -0.05),
                'sharpe_ratio': np.random.normal(1.2, 0.2)
            })

        intervals = self.validator.confidence_intervals(mc_results, confidence_level=0.95)

        assert 'total_return' in intervals
        assert 'max_drawdown' in intervals
        assert 'sharpe_ratio' in intervals

        # Verificar que intervalos fazem sentido
        for metric, (lower, upper) in intervals.items():
            assert lower < upper
            assert isinstance(lower, (int, float))
            assert isinstance(upper, (int, float))