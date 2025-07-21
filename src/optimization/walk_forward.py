import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta

from .parameter_optimizer import ParameterOptimizer
from ..backtest.backtest_engine import BacktestEngine


class WalkForwardAnalyzer:
    """Analisador Walk-Forward para validação temporal"""

    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def split_data_for_walk_forward(self, data: pd.DataFrame, n_windows: int,
                                    optimization_length_pct: float = 0.7) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Divide dados para análise walk-forward"""

        total_length = len(data)

        # Calcular tamanhos
        optimization_length = int(total_length * optimization_length_pct / n_windows)
        test_length = int(total_length * (1 - optimization_length_pct) / n_windows)

        # Garantir que temos dados suficientes
        min_optimization_length = 100  # Mínimo para otimização
        min_test_length = 30  # Mínimo para teste

        if optimization_length < min_optimization_length:
            optimization_length = min_optimization_length
        if test_length < min_test_length:
            test_length = min_test_length

        window_pairs = []

        for i in range(n_windows):
            # Calcular índices da janela
            start_idx = i * (optimization_length + test_length)

            if start_idx + optimization_length + test_length > total_length:
                break

            opt_end_idx = start_idx + optimization_length
            test_end_idx = opt_end_idx + test_length

            # Extrair dados
            optimization_data = data.iloc[start_idx:opt_end_idx].copy()
            test_data = data.iloc[opt_end_idx:test_end_idx].copy()

            if len(optimization_data) >= min_optimization_length and len(test_data) >= min_test_length:
                window_pairs.append((optimization_data, test_data))

        logging.info(f"📊 Criadas {len(window_pairs)} janelas walk-forward")
        logging.info(f"   Tamanho otimização: ~{optimization_length} registros")
        logging.info(f"   Tamanho teste: ~{test_length} registros")

        return window_pairs

    def run_walk_forward_analysis(self, data: pd.DataFrame, parameter_space: Dict[str, List[Any]],
                                  n_windows: int = 4, optimization_length_pct: float = 0.7,
                                  max_combinations_per_window: int = 100) -> List[Dict[str, Any]]:
        """Executa análise walk-forward completa"""

        logging.info("📈 Iniciando análise walk-forward")

        # Dividir dados
        window_pairs = self.split_data_for_walk_forward(data, n_windows, optimization_length_pct)

        if not window_pairs:
            raise ValueError("Não foi possível criar janelas válidas para walk-forward")

        results = []

        for i, (opt_data, test_data) in enumerate(window_pairs, 1):
            logging.info(f"\n🔄 Processando janela {i}/{len(window_pairs)}")
            logging.info(f"   Otimização: {opt_data['timestamp'].min()} até {opt_data['timestamp'].max()}")
            logging.info(f"   Teste: {test_data['timestamp'].min()} até {test_data['timestamp'].max()}")

            try:
                # Otimizar parâmetros na janela de otimização
                optimizer = ParameterOptimizer(self.initial_capital, self.commission)

                best_params, opt_results = optimizer.optimize_parameters(
                    data=opt_data,
                    parameter_space=parameter_space,
                    max_combinations=max_combinations_per_window,
                    n_jobs=2  # Menos paralelismo para não sobrecarregar
                )

                # Ao rodar o backtest na janela de teste, mesclar STRATEGY_CONFIG.__dict__ com best_params antes de passar para o BacktestEngine.
                # Exemplo:
                # config = STRATEGY_CONFIG.__dict__.copy()
                # config.update(best_params)
                # test_result = test_engine.run_backtest(test_data, config)
                # ...
                # Testar na janela de teste (out-of-sample)
                test_engine = BacktestEngine(self.initial_capital, self.commission)
                test_result = test_engine.run_backtest(test_data, best_params)

                # Compilar resultados
                window_result = {
                    'window': i,
                    'optimization_start': opt_data['timestamp'].min(),
                    'optimization_end': opt_data['timestamp'].max(),
                    'test_start': test_data['timestamp'].min(),
                    'test_end': test_data['timestamp'].max(),
                    'optimization_records': len(opt_data),
                    'test_records': len(test_data),

                    # Melhores parâmetros encontrados
                    **{f'best_{k}': v for k, v in best_params.items()},

                    # Resultados in-sample (otimização)
                    'in_sample_return': max([r['total_return'] for r in opt_results]),
                    'in_sample_drawdown': min([r['max_drawdown'] for r in opt_results]),
                    'in_sample_sharpe': max([r['sharpe_ratio'] for r in opt_results]),

                    # Resultados out-of-sample (teste)
                    'out_sample_return': test_result['total_return'],
                    'out_sample_drawdown': test_result['max_drawdown'],
                    'out_sample_sharpe': test_result['sharpe_ratio'],
                    'out_sample_trades': test_result['total_trades'],
                    'out_sample_win_rate': test_result['win_rate'],
                    'out_sample_profit_factor': test_result['profit_factor'],
                    'out_sample_expectancy': test_result['expectancy'],

                    # Métricas de degradação
                    'return_degradation': test_result['total_return'] - max([r['total_return'] for r in opt_results]),
                    'sharpe_degradation': test_result['sharpe_ratio'] - max([r['sharpe_ratio'] for r in opt_results])
                }

                results.append(window_result)

                logging.info(f"✅ Janela {i} concluída:")
                logging.info(f"   In-sample: {window_result['in_sample_return']:.2%}")
                logging.info(f"   Out-sample: {window_result['out_sample_return']:.2%}")
                logging.info(f"   Degradação: {window_result['return_degradation']:.2%}")

            except Exception as e:
                logging.error(f"❌ Erro na janela {i}: {e}")
                continue

        logging.info(f"\n📊 Walk-forward concluído: {len(results)} janelas processadas")

        return results

    def analyze_consistency(self, wf_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa consistência dos resultados walk-forward"""

        if not wf_results:
            return {'error': 'Nenhum resultado para analisar'}

        df = pd.DataFrame(wf_results)

        # Estatísticas básicas
        out_sample_returns = df['out_sample_return']
        out_sample_drawdowns = df['out_sample_drawdown']
        out_sample_sharpes = df['out_sample_sharpe']

        # Análise de consistência
        positive_periods = (out_sample_returns > 0).sum()
        total_periods = len(out_sample_returns)
        consistency_ratio = positive_periods / total_periods

        # Análise de degradação
        return_degradations = df['return_degradation']
        sharpe_degradations = df['sharpe_degradation']

        # Estatísticas de estabilidade
        return_stability = 1 - (
                    out_sample_returns.std() / abs(out_sample_returns.mean())) if out_sample_returns.mean() != 0 else 0

        # Análise de parâmetros
        param_columns = [col for col in df.columns if col.startswith('best_')]
        parameter_stability = {}

        for param_col in param_columns:
            param_name = param_col.replace('best_', '')
            unique_values = df[param_col].nunique()
            total_windows = len(df)
            stability = 1 - (unique_values / total_windows)
            parameter_stability[param_name] = stability

        # Métricas de risco
        worst_drawdown = out_sample_drawdowns.min()
        avg_drawdown = out_sample_drawdowns.mean()

        return {
            # Retornos
            'avg_return': out_sample_returns.mean(),
            'std_return': out_sample_returns.std(),
            'min_return': out_sample_returns.min(),
            'max_return': out_sample_returns.max(),
            'median_return': out_sample_returns.median(),

            # Consistência
            'positive_periods': positive_periods,
            'total_periods': total_periods,
            'consistency_ratio': consistency_ratio,
            'return_stability': max(0, return_stability),

            # Sharpe
            'avg_sharpe': out_sample_sharpes.mean(),
            'std_sharpe': out_sample_sharpes.std(),
            'min_sharpe': out_sample_sharpes.min(),
            'max_sharpe': out_sample_sharpes.max(),

            # Drawdown
            'avg_drawdown': avg_drawdown,
            'worst_drawdown': worst_drawdown,
            'std_drawdown': out_sample_drawdowns.std(),

            # Degradação
            'avg_return_degradation': return_degradations.mean(),
            'avg_sharpe_degradation': sharpe_degradations.mean(),
            'worst_return_degradation': return_degradations.min(),

            # Estabilidade de parâmetros
            'parameter_stability': parameter_stability,
            'avg_parameter_stability': np.mean(list(parameter_stability.values())),

            # Classificação geral
            'overall_grade': self._calculate_overall_grade(
                consistency_ratio, return_stability, avg_drawdown,
                out_sample_returns.mean(), return_degradations.mean()
            )
        }

    def _calculate_overall_grade(self, consistency_ratio: float, return_stability: float,
                                 avg_drawdown: float, avg_return: float, avg_degradation: float) -> str:
        """Calcula nota geral da estratégia"""

        score = 0

        # Consistência (0-30 pontos)
        if consistency_ratio >= 0.8:
            score += 30
        elif consistency_ratio >= 0.6:
            score += 20
        elif consistency_ratio >= 0.4:
            score += 10

        # Estabilidade de retorno (0-25 pontos)
        if return_stability >= 0.7:
            score += 25
        elif return_stability >= 0.5:
            score += 15
        elif return_stability >= 0.3:
            score += 8

        # Drawdown (0-20 pontos)
        if avg_drawdown >= -0.1:
            score += 20
        elif avg_drawdown >= -0.2:
            score += 15
        elif avg_drawdown >= -0.3:
            score += 10
        elif avg_drawdown >= -0.4:
            score += 5

        # Retorno médio (0-15 pontos)
        if avg_return >= 0.2:
            score += 15
        elif avg_return >= 0.1:
            score += 12
        elif avg_return >= 0.05:
            score += 8
        elif avg_return >= 0:
            score += 4

        # Degradação (0-10 pontos)
        if avg_degradation >= -0.05:
            score += 10
        elif avg_degradation >= -0.1:
            score += 7
        elif avg_degradation >= -0.2:
            score += 4

        # Classificar
        if score >= 85:
            return "A+ (Excelente)"
        elif score >= 75:
            return "A (Muito Bom)"
        elif score >= 65:
            return "B+ (Bom)"
        elif score >= 55:
            return "B (Regular)"
        elif score >= 45:
            return "C+ (Abaixo da Média)"
        elif score >= 35:
            return "C (Ruim)"
        else:
            return "D (Muito Ruim)"

    def generate_walk_forward_report(self, wf_results: List[Dict[str, Any]],
                                     consistency_metrics: Dict[str, Any]) -> str:
        """Gera relatório detalhado da análise walk-forward"""

        report = []
        report.append("📈 RELATÓRIO DE ANÁLISE WALK-FORWARD")
        report.append("=" * 60)

        # Resumo geral
        report.append(f"\n📊 RESUMO GERAL:")
        report.append(f"   Janelas analisadas: {consistency_metrics['total_periods']}")
        report.append(f"   Períodos positivos: {consistency_metrics['positive_periods']}")
        report.append(f"   Taxa de consistência: {consistency_metrics['consistency_ratio']:.1%}")
        report.append(f"   Nota geral: {consistency_metrics['overall_grade']}")

        # Métricas de performance
        report.append(f"\n💰 PERFORMANCE OUT-OF-SAMPLE:")
        report.append(f"   Retorno médio: {consistency_metrics['avg_return']:.2%}")
        report.append(f"   Retorno médio: {consistency_metrics['avg_return']:.2%}")
        report.append(f"   Desvio padrão: {consistency_metrics['std_return']:.2%}")
        report.append(f"   Melhor período: {consistency_metrics['max_return']:.2%}")
        report.append(f"   Pior período: {consistency_metrics['min_return']:.2%}")
        report.append(f"   Mediana: {consistency_metrics['median_return']:.2%}")

        # Métricas de risco
        report.append(f"\n📉 ANÁLISE DE RISCO:")
        report.append(f"   Drawdown médio: {consistency_metrics['avg_drawdown']:.2%}")
        report.append(f"   Pior drawdown: {consistency_metrics['worst_drawdown']:.2%}")
        report.append(f"   Sharpe médio: {consistency_metrics['avg_sharpe']:.3f}")
        report.append(f"   Estabilidade de retorno: {consistency_metrics['return_stability']:.1%}")

        # Análise de degradação
        report.append(f"\n⚠️ ANÁLISE DE DEGRADAÇÃO:")
        report.append(f"   Degradação média de retorno: {consistency_metrics['avg_return_degradation']:.2%}")
        report.append(f"   Pior degradação: {consistency_metrics['worst_return_degradation']:.2%}")
        report.append(f"   Degradação média Sharpe: {consistency_metrics['avg_sharpe_degradation']:.3f}")

        # Estabilidade de parâmetros
        report.append(f"\n🔧 ESTABILIDADE DE PARÂMETROS:")
        report.append(f"   Estabilidade média: {consistency_metrics['avg_parameter_stability']:.1%}")

        param_stability = consistency_metrics['parameter_stability']
        for param, stability in sorted(param_stability.items(), key=lambda x: x[1], reverse=True):
            report.append(f"   {param}: {stability:.1%}")

        # Detalhes por janela
        report.append(f"\n📋 DETALHES POR JANELA:")
        report.append("-" * 60)

        for result in wf_results:
            report.append(f"\nJanela {result['window']}:")
            report.append(
                f"   Período: {result['test_start'].strftime('%Y-%m-%d')} até {result['test_end'].strftime('%Y-%m-%d')}")
            report.append(f"   Retorno: {result['out_sample_return']:.2%}")
            report.append(f"   Drawdown: {result['out_sample_drawdown']:.2%}")
            report.append(f"   Trades: {result['out_sample_trades']}")
            report.append(f"   Win Rate: {result['out_sample_win_rate']:.1%}")
            report.append(f"   Degradação: {result['return_degradation']:.2%}")

        # Recomendações
        report.append(f"\n💡 RECOMENDAÇÕES:")

        if consistency_metrics['consistency_ratio'] >= 0.7:
            report.append("   ✅ Estratégia demonstra boa consistência temporal")
        else:
            report.append("   ⚠️ Estratégia pode ser instável em diferentes períodos")

        if consistency_metrics['avg_return_degradation'] >= -0.1:
            report.append("   ✅ Baixa degradação entre in-sample e out-of-sample")
        else:
            report.append("   ⚠️ Alta degradação - possível overfitting")

        if consistency_metrics['worst_drawdown'] >= -0.25:
            report.append("   ✅ Drawdown controlado")
        else:
            report.append("   ⚠️ Drawdown alto - considere ajustar gestão de risco")

        if consistency_metrics['avg_parameter_stability'] >= 0.6:
            report.append("   ✅ Parâmetros relativamente estáveis")
        else:
            report.append("   ⚠️ Parâmetros muito variáveis - estratégia pode ser sensível")

        return "\n".join(report)