import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch

from src.data.data_manager import DataManager
from src.backtest.backtest_engine import BacktestEngine
from src.optimization.parameter_optimizer import ParameterOptimizer


class TestIntegration:
    """Testes de integração do sistema completo"""

    def setup_method(self):
        """Setup para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(cache_dir=self.temp_dir)
        self.engine = BacktestEngine(initial_capital=10000)
        self.optimizer = ParameterOptimizer(initial_capital=10000)

    def test_complete_backtest_workflow(self, sample_data, sample_config):
        """Testa fluxo completo de backtest"""
        # 1. Simular carregamento de dados
        with patch.object(self.data_manager, 'load_data', return_value=sample_data):
            data = self.data_manager.load_data('BTCBRL', '1h', 180)

        # 2. Executar backtest
        results = self.engine.run_backtest(data, sample_config)

        # 3. Verificar resultados
        assert results is not None
        assert 'total_return' in results
        assert 'equity_curve' in results
        assert isinstance(results['equity_curve'], pd.DataFrame)

        # 4. Verificar que equity curve tem estrutura correta
        equity_curve = results['equity_curve']
        assert 'timestamp' in equity_curve.columns
        assert 'equity' in equity_curve.columns
        assert len(equity_curve) > 0

    def test_optimization_workflow(self, sample_data, parameter_space):
        """Testa fluxo completo de otimização"""
        # 1. Executar otimização
        best_params, optimization_results = self.optimizer.optimize_parameters(
            data=sample_data,
            parameter_space=parameter_space,
            max_combinations=10,  # Pequeno para teste rápido
            n_jobs=1
        )

        # 2. Verificar resultados da otimização
        assert isinstance(best_params, dict)
        assert isinstance(optimization_results, list)
        assert len(optimization_results) > 0

        # 3. Verificar que melhores parâmetros são válidos
        for key, value in best_params.items():
            assert key.replace('param_', '') in parameter_space
            # Note: valor pode não estar na lista original devido à amostragem

        # 4. Testar melhores parâmetros em backtest
        final_results = self.engine.run_backtest(sample_data, best_params)
        assert final_results is not None
        assert 'total_return' in final_results

    def test_data_pipeline_integration(self, sample_data):
        """Testa integração do pipeline de dados"""
        # 1. Simular salvamento no cache
        cache_file = self.data_manager._get_cache_filename('BTCBRL', '1h', 180)
        sample_data.to_parquet(cache_file, index=False)

        # 2. Carregar do cache
        cached_data = self.data_manager._load_from_cache('BTCBRL', '1h', 180)

        # 3. Verificar integridade
        assert cached_data is not None
        assert len(cached_data) == len(sample_data)
        pd.testing.assert_frame_equal(cached_data, sample_data)

        # 4. Usar dados em backtest
        config = {
            'fast_ma': 9, 'slow_ma': 21, 'rsi_period': 14,
            'take_profit_pct': 0.025, 'stop_loss_pct': 0.015, 'min_score': 60
        }

        results = self.engine.run_backtest(cached_data, config)
        assert results is not None

    def test_error_handling_integration(self):
        """Testa tratamento de erros em integração"""
        # 1. Dados inválidos
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'close': [np.nan] * 10  # Todos NaN
        })

        config = {'fast_ma': 9, 'slow_ma': 21, 'min_score': 60}

        # Deve lidar graciosamente com dados inválidos
        results = self.engine.run_backtest(invalid_data, config)
        assert results['total_trades'] == 0

        # 2. Configuração inválida
        invalid_config = {'fast_ma': 'invalid', 'slow_ma': 21}

        # Deve lidar com configuração inválida
        with pytest.raises((ValueError, TypeError, KeyError)):
            self.engine.run_backtest(sample_data, invalid_config)

    def test_performance_integration(self, sample_data):
        """Testa performance do sistema integrado"""
        import time

        config = {
            'fast_ma': 9, 'slow_ma': 21, 'rsi_period': 14,
            'take_profit_pct': 0.025, 'stop_loss_pct': 0.015, 'min_score': 60
        }

        # Medir tempo de backtest
        start_time = time.time()
        results = self.engine.run_backtest(sample_data, config)
        backtest_time = time.time() - start_time

        # Backtest deve ser rápido (< 5 segundos para dados de teste)
        assert backtest_time < 5.0
        assert results is not None

        # Medir tempo de otimização pequena
        parameter_space = {
            'fast_ma': [7, 9, 12],
            'slow_ma': [18, 21, 26]
        }

        start_time = time.time()
        best_params, _ = self.optimizer.optimize_parameters(
            sample_data, parameter_space, max_combinations=9, n_jobs=1
        )
        optimization_time = time.time() - start_time

        # Otimização pequena deve ser razoavelmente rápida (< 30 segundos)
        assert optimization_time < 30.0
        assert best_params is not None

        def test_memory_usage_integration(self, sample_data):
            """Testa uso de memória do sistema integrado"""
            import psutil
            import os

            # Medir memória inicial
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Executar múltiplos backtests
            config = {
                'fast_ma': 9, 'slow_ma': 21, 'rsi_period': 14,
                'take_profit_pct': 0.025, 'stop_loss_pct': 0.015, 'min_score': 60
            }

            results_list = []
            for i in range(10):
                results = self.engine.run_backtest(sample_data, config)
                results_list.append(results)
                self.engine.reset()  # Reset para próximo backtest

            # Medir memória final
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Aumento de memória deve ser razoável (< 100MB para testes)
            assert memory_increase < 100
            assert len(results_list) == 10

        def test_concurrent_execution(self, sample_data):
            """Testa execução concorrente (simulada)"""
            from concurrent.futures import ThreadPoolExecutor
            import threading

            configs = [
                {'fast_ma': 5, 'slow_ma': 15, 'min_score': 60},
                {'fast_ma': 9, 'slow_ma': 21, 'min_score': 65},
                {'fast_ma': 12, 'slow_ma': 26, 'min_score': 70}
            ]

            def run_backtest_thread(config):
                # Cada thread precisa de sua própria instância
                engine = BacktestEngine(initial_capital=10000)
                return engine.run_backtest(sample_data, config)

            # Executar em paralelo
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(run_backtest_thread, config) for config in configs]
                results = [future.result() for future in futures]

            # Verificar que todos os backtests foram executados
            assert len(results) == 3
            for result in results:
                assert result is not None
                assert 'total_return' in result

        def test_data_consistency_across_modules(self, sample_data):
            """Testa consistência de dados entre módulos"""
            # 1. Processar dados com indicadores
            from indicators.technical_indicators import TechnicalIndicators

            config = {
                'fast_ma': 9, 'slow_ma': 21, 'rsi_period': 14,
                'take_profit_pct': 0.025, 'stop_loss_pct': 0.015, 'min_score': 60
            }

            data_with_indicators = TechnicalIndicators.calculate_all_indicators(
                sample_data, config
            )

            # 2. Usar em backtest
            results = self.engine.run_backtest(sample_data, config)

            # 3. Verificar que dados são consistentes
            assert len(data_with_indicators) == len(sample_data)
            assert 'ema_fast' in data_with_indicators.columns
            assert 'rsi' in data_with_indicators.columns

            # 4. Verificar que backtest processou todos os dados
            equity_curve = results['equity_curve']
            assert len(equity_curve) <= len(sample_data)  # Pode ser menor devido a período de aquecimento

    class TestEndToEnd:
        """Testes end-to-end do sistema completo"""

        def test_complete_trading_system_simulation(self, sample_data):
            """Simula uso completo do sistema de trading"""
            # 1. Setup inicial
            data_manager = DataManager()
            engine = BacktestEngine(initial_capital=10000)
            optimizer = ParameterOptimizer(initial_capital=10000)

            # 2. Configuração inicial
            initial_config = {
                'fast_ma': 9, 'slow_ma': 21, 'rsi_period': 14,
                'rsi_oversold': 30, 'rsi_overbought': 70,
                'take_profit_pct': 0.025, 'stop_loss_pct': 0.015,
                'volume_threshold': 1.2, 'min_score': 60
            }

            # 3. Backtest inicial
            initial_results = engine.run_backtest(sample_data, initial_config)
            assert initial_results is not None

            # 4. Se resultados não são satisfatórios, otimizar
            if initial_results['sharpe_ratio'] < 1.0:
                parameter_space = {
                    'fast_ma': [7, 9, 12],
                    'slow_ma': [18, 21, 26],
                    'take_profit_pct': [0.02, 0.025, 0.03],
                    'min_score': [55, 60, 65]
                }

                best_params, _ = optimizer.optimize_parameters(
                    sample_data, parameter_space, max_combinations=20, n_jobs=1
                )

                # 5. Backtest com parâmetros otimizados
                optimized_results = engine.run_backtest(sample_data, best_params)
                assert optimized_results is not None

                # 6. Comparar resultados
                # Resultados otimizados devem ser pelo menos tão bons quanto iniciais
                # (nem sempre verdade devido à aleatoriedade, mas testamos a estrutura)
                assert 'total_return' in optimized_results
                assert 'sharpe_ratio' in optimized_results

            # 7. Gerar relatório final
            final_results = optimized_results if 'optimized_results' in locals() else initial_results

            assert final_results['total_trades'] >= 0
            assert isinstance(final_results['equity_curve'], pd.DataFrame)
            assert isinstance(final_results['trades_detail'], pd.DataFrame)

        def test_system_robustness(self, sample_data):
            """Testa robustez do sistema com dados diversos"""
            engine = BacktestEngine(initial_capital=10000)

            # Teste com diferentes configurações
            test_configs = [
                # Configuração conservadora
                {
                    'fast_ma': 12, 'slow_ma': 26, 'rsi_period': 14,
                    'take_profit_pct': 0.015, 'stop_loss_pct': 0.01, 'min_score': 70
                },
                # Configuração agressiva
                {
                    'fast_ma': 5, 'slow_ma': 15, 'rsi_period': 10,
                    'take_profit_pct': 0.04, 'stop_loss_pct': 0.02, 'min_score': 50
                },
                # Configuração balanceada
                {
                    'fast_ma': 9, 'slow_ma': 21, 'rsi_period': 14,
                    'take_profit_pct': 0.025, 'stop_loss_pct': 0.015, 'min_score': 60
                }
            ]

            results_list = []
            for config in test_configs:
                try:
                    results = engine.run_backtest(sample_data, config)
                    results_list.append(results)
                    engine.reset()
                except Exception as e:
                    pytest.fail(f"Sistema falhou com configuração {config}: {e}")

            # Verificar que todas as configurações foram processadas
            assert len(results_list) == len(test_configs)

            # Verificar consistência dos resultados
            for results in results_list:
                assert results is not None
                assert 'total_return' in results
                assert 'final_capital' in results
                assert results['final_capital'] > 0  # Capital nunca deve ser negativo

        def test_system_scalability(self):
            """Testa escalabilidade do sistema"""
            # Gerar dataset maior
            large_data = pd.DataFrame({
                'timestamp': pd.date_range('2020-01-01', periods=10000, freq='1H'),
                'open': np.random.uniform(45000, 55000, 10000),
                'high': np.random.uniform(45000, 55000, 10000),
                'low': np.random.uniform(45000, 55000, 10000),
                'close': np.random.uniform(45000, 55000, 10000),
                'volume': np.random.uniform(100, 1000, 10000)
            })

            # Corrigir OHLC
            large_data['high'] = np.maximum(large_data['high'],
                                            np.maximum(large_data['open'], large_data['close']))
            large_data['low'] = np.minimum(large_data['low'],
                                           np.minimum(large_data['open'], large_data['close']))

            engine = BacktestEngine(initial_capital=10000)
            config = {
                'fast_ma': 9, 'slow_ma': 21, 'rsi_period': 14,
                'take_profit_pct': 0.025, 'stop_loss_pct': 0.015, 'min_score': 60
            }

            import time
            start_time = time.time()
            results = engine.run_backtest(large_data, config)
            execution_time = time.time() - start_time

            # Sistema deve processar 10k registros em tempo razoável (< 30 segundos)
            assert execution_time < 30
            assert results is not None
            assert results['total_trades'] >= 0