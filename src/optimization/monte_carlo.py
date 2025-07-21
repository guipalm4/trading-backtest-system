import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import random
from tqdm import tqdm

from ..backtest.backtest_engine import BacktestEngine


class MonteCarloValidator:
    """Validador Monte Carlo para análise de robustez"""

    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission

    def bootstrap_data(self, data: pd.DataFrame, sample_ratio: float = 0.8) -> pd.DataFrame:
        """Cria amostra bootstrap dos dados"""

        n_samples = int(len(data) * sample_ratio)

        # Amostragem com reposição
        sampled_indices = np.random.choice(len(data), size=n_samples, replace=True)
        sampled_data = data.iloc[sampled_indices].copy()

        # Reordenar por timestamp para manter sequência temporal
        sampled_data = sampled_data.sort_values('timestamp').reset_index(drop=True)

        return sampled_data

    def block_bootstrap_data(self, data: pd.DataFrame, block_size: int = 20,
                             sample_ratio: float = 0.8) -> pd.DataFrame:
        """Cria amostra usando block bootstrap (preserva dependência temporal)"""

        n_samples = int(len(data) * sample_ratio)
        n_blocks = n_samples // block_size

        sampled_data = []

        for _ in range(n_blocks):
            # Escolher início aleatório do bloco
            max_start = len(data) - block_size
            if max_start <= 0:
                block_start = 0
                current_block_size = len(data)
            else:
                block_start = random.randint(0, max_start)
                current_block_size = min(block_size, len(data) - block_start)

            # Extrair bloco
            block = data.iloc[block_start:block_start + current_block_size].copy()
            sampled_data.append(block)

        # Concatenar blocos
        if sampled_data:
            result = pd.concat(sampled_data, ignore_index=True)
            # Reordenar por timestamp
            result = result.sort_values('timestamp').reset_index(drop=True)
            return result
        else:
            return data.copy()

    def noise_injection_data(self, data: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
        """Adiciona ruído aos dados para testar robustez"""

        noisy_data = data.copy()

        # Adicionar ruído aos preços
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in noisy_data.columns:
                noise = np.random.normal(0, noise_level, len(noisy_data))
                noisy_data[col] = noisy_data[col] * (1 + noise)

        # Adicionar ruído ao volume
        if 'volume' in noisy_data.columns:
            volume_noise = np.random.normal(0, noise_level * 2, len(noisy_data))  # Mais ruído no volume
            noisy_data['volume'] = noisy_data['volume'] * (1 + volume_noise)
            noisy_data['volume'] = noisy_data['volume'].clip(lower=0)  # Volume não pode ser negativo

        return noisy_data

    def run_monte_carlo_simulation(self, data: pd.DataFrame, config: Dict[str, Any],
                                   n_simulations: int = 1000, sample_ratio: float = 0.8,
                                   use_block_bootstrap: bool = True,
                                   add_noise: bool = True) -> List[Dict[str, Any]]:
        """Executa simulação Monte Carlo completa"""

        logging.info(f"🎲 Iniciando simulação Monte Carlo: {n_simulations} simulações")

        results = []

        for i in tqdm(range(n_simulations), desc="Executando simulações Monte Carlo"):
            try:
                # Criar amostra dos dados
                if use_block_bootstrap:
                    sampled_data = self.block_bootstrap_data(data, sample_ratio=sample_ratio)
                else:
                    sampled_data = self.bootstrap_data(data, sample_ratio=sample_ratio)

                # Adicionar ruído se solicitado
                if add_noise:
                    sampled_data = self.noise_injection_data(sampled_data, noise_level=0.005)

                # Executar backtest
                # Ao rodar o backtest, use config = STRATEGY_CONFIG.__dict__.copy(); config.update(config_param) se necessário, antes de passar para o BacktestEngine.
                engine = BacktestEngine(self.initial_capital, self.commission)
                result = engine.run_backtest(sampled_data, config)

                # Adicionar informações da simulação
                simulation_result = {
                    'simulation': i + 1,
                    'sample_size': len(sampled_data),
                    'total_return': result['total_return'],
                    'max_drawdown': result['max_drawdown'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'total_trades': result['total_trades'],
                    'win_rate': result['win_rate'],
                    'profit_factor': result['profit_factor'],
                    'expectancy': result['expectancy'],
                    'net_profit': result['net_profit'],
                    'largest_win': result['largest_win'],
                    'largest_loss': result['largest_loss'],
                    'avg_trade_duration': result['avg_trade_duration'].total_seconds() / 3600 if result[
                        'avg_trade_duration'] else 0  # em horas
                }

                results.append(simulation_result)

            except Exception as e:
                logging.warning(f"Erro na simulação {i + 1}: {e}")
                continue

        logging.info(f"✅ Monte Carlo concluído: {len(results)} simulações válidas")

        return results

    def calculate_statistics(self, mc_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula estatísticas dos resultados Monte Carlo"""

        if not mc_results:
            return {'error': 'Nenhum resultado para analisar'}

        df = pd.DataFrame(mc_results)

        # Estatísticas de retorno
        returns = df['total_return']

        statistics = {
            # Estatísticas básicas de retorno
            'mean_return': returns.mean(),
            'median_return': returns.median(),
            'std_return': returns.std(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),

            # Percentis
            'percentile_5': returns.quantile(0.05),
            'percentile_10': returns.quantile(0.10),
            'percentile_25': returns.quantile(0.25),
            'percentile_75': returns.quantile(0.75),
            'percentile_90': returns.quantile(0.90),
            'percentile_95': returns.quantile(0.95),

            # Value at Risk e Conditional VaR
            'var_95': returns.quantile(0.05),  # VaR 95%
            'var_99': returns.quantile(0.01),  # VaR 99%
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),  # CVaR 95%
            'cvar_99': returns[returns <= returns.quantile(0.01)].mean(),  # CVaR 99%

            # Probabilidades
            'probability_profit': (returns > 0).mean(),
            'probability_loss_5pct': (returns < -0.05).mean(),
            'probability_loss_10pct': (returns < -0.10).mean(),
            'probability_loss_20pct': (returns < -0.20).mean(),
            'probability_gain_10pct': (returns > 0.10).mean(),
            'probability_gain_20pct': (returns > 0.20).mean(),
            'probability_gain_50pct': (returns > 0.50).mean(),

            # Estatísticas de drawdown
            'mean_drawdown': df['max_drawdown'].mean(),
            'worst_drawdown': df['max_drawdown'].min(),
            'std_drawdown': df['max_drawdown'].std(),
            'probability_drawdown_20pct': (df['max_drawdown'] < -0.20).mean(),
            'probability_drawdown_30pct': (df['max_drawdown'] < -0.30).mean(),

            # Estatísticas de Sharpe
            'mean_sharpe': df['sharpe_ratio'].mean(),
            'median_sharpe': df['sharpe_ratio'].median(),
            'std_sharpe': df['sharpe_ratio'].std(),
            'probability_positive_sharpe': (df['sharpe_ratio'] > 0).mean(),
            'probability_sharpe_above_1': (df['sharpe_ratio'] > 1).mean(),

            # Estatísticas de trades
            'mean_total_trades': df['total_trades'].mean(),
            'mean_win_rate': df['win_rate'].mean(),
            'mean_profit_factor': df['profit_factor'].mean(),
            'mean_expectancy': df['expectancy'].mean(),

            # Estabilidade
            'return_stability': 1 - (returns.std() / abs(returns.mean())) if returns.mean() != 0 else 0,
            'coefficient_of_variation': returns.std() / abs(returns.mean()) if returns.mean() != 0 else float('inf'),

            # Métricas de risco-retorno
            'return_to_risk_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'downside_deviation': returns[returns < 0].std(),
            'sortino_ratio': returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0,

            # Informações da simulação
            'total_simulations': len(mc_results),
            'valid_simulations': len(df[df['total_trades'] > 0]),
            'simulation_success_rate': len(df[df['total_trades'] > 0]) / len(df)
        }

        return statistics

    def analyze_parameter_robustness(self, data: pd.DataFrame, base_config: Dict[str, Any],
                                     parameter_variations: Dict[str, List[Any]],
                                     n_simulations_per_variation: int = 100) -> Dict[str, Dict]:
        """Analisa robustez dos parâmetros usando Monte Carlo"""

        logging.info("🔬 Analisando robustez de parâmetros")

        robustness_results = {}

        for param_name, param_values in parameter_variations.items():
            logging.info(f"   Testando parâmetro: {param_name}")

            param_results = {}

            for param_value in param_values:
                # Criar configuração com parâmetro variado
                test_config = base_config.copy()
                test_config[param_name] = param_value

                # Executar Monte Carlo para esta configuração
                mc_results = self.run_monte_carlo_simulation(
                    data=data,
                    config=test_config,
                    n_simulations=n_simulations_per_variation,
                    sample_ratio=0.7  # Menor para ser mais rápido
                )

                if mc_results:
                    # Calcular estatísticas resumidas
                    returns = [r['total_return'] for r in mc_results]
                    drawdowns = [r['max_drawdown'] for r in mc_results]

                    param_results[param_value] = {
                        'mean_return': np.mean(returns),
                        'std_return': np.std(returns),
                        'probability_profit': sum(1 for r in returns if r > 0) / len(returns),
                        'var_95': np.percentile(returns, 5),
                        'mean_drawdown': np.mean(drawdowns),
                        'worst_drawdown': min(drawdowns),
                        'simulations_count': len(mc_results)
                    }

            robustness_results[param_name] = param_results

        return robustness_results

    def generate_monte_carlo_report(self, mc_results: List[Dict[str, Any]],
                                    statistics: Dict[str, Any]) -> str:
        """Gera relatório detalhado da análise Monte Carlo"""

        report = []
        report.append("🎲 RELATÓRIO DE ANÁLISE MONTE CARLO")
        report.append("=" * 60)

        # Informações da simulação
        report.append(f"\n📊 INFORMAÇÕES DA SIMULAÇÃO:")
        report.append(f"   Total de simulações: {statistics['total_simulations']}")
        report.append(f"   Simulações válidas: {statistics['valid_simulations']}")
        report.append(f"   Taxa de sucesso: {statistics['simulation_success_rate']:.1%}")

        # Estatísticas de retorno
        report.append(f"\n💰 ESTATÍSTICAS DE RETORNO:")
        report.append(f"   Retorno médio: {statistics['mean_return']:.2%}")
        report.append(f"   Retorno mediano: {statistics['median_return']:.2%}")
        report.append(f"   Desvio padrão: {statistics['std_return']:.2%}")
        report.append(f"   Melhor caso: {statistics['max_return']:.2%}")
        report.append(f"   Pior caso: {statistics['min_return']:.2%}")

        # Distribuição de probabilidades
        report.append(f"\n🎯 PROBABILIDADES:")
        report.append(f"   Probabilidade de lucro: {statistics['probability_profit']:.1%}")
        report.append(f"   Prob. ganho > 10%: {statistics['probability_gain_10pct']:.1%}")
        report.append(f"   Prob. ganho > 20%: {statistics['probability_gain_20pct']:.1%}")
        report.append(f"   Prob. perda > 10%: {statistics['probability_loss_10pct']:.1%}")
        report.append(f"   Prob. perda > 20%: {statistics['probability_loss_20pct']:.1%}")

        # Value at Risk
        report.append(f"\n⚠️ ANÁLISE DE RISCO (VaR):")
        report.append(f"   VaR 95%: {statistics['var_95']:.2%}")
        report.append(f"   VaR 99%: {statistics['var_99']:.2%}")
        report.append(f"   CVaR 95%: {statistics['cvar_95']:.2%}")
        report.append(f"   CVaR 99%: {statistics['cvar_99']:.2%}")

        # Percentis
        report.append(f"\n📈 DISTRIBUIÇÃO DE RETORNOS:")
        report.append(f"   5º percentil: {statistics['percentile_5']:.2%}")
        report.append(f"   25º percentil: {statistics['percentile_25']:.2%}")
        report.append(f"   75º percentil: {statistics['percentile_75']:.2%}")
        report.append(f"   95º percentil: {statistics['percentile_95']:.2%}")

        # Drawdown
        report.append(f"\n📉 ANÁLISE DE DRAWDOWN:")
        report.append(f"   Drawdown médio: {statistics['mean_drawdown']:.2%}")
        report.append(f"   Pior drawdown: {statistics['worst_drawdown']:.2%}")
        report.append(f"   Prob. drawdown > 20%: {statistics['probability_drawdown_20pct']:.1%}")
        report.append(f"   Prob. drawdown > 30%: {statistics['probability_drawdown_30pct']:.1%}")

        # Métricas de qualidade
        report.append(f"\n⚡ MÉTRICAS DE QUALIDADE:")
        report.append(f"   Sharpe médio: {statistics['mean_sharpe']:.3f}")
        report.append(f"   Prob. Sharpe > 0: {statistics['probability_positive_sharpe']:.1%}")
        report.append(f"   Prob. Sharpe > 1: {statistics['probability_sharpe_above_1']:.1%}")
        report.append(f"   Sortino ratio: {statistics['sortino_ratio']:.3f}")

        # Estabilidade
        report.append(f"\n🔄 ESTABILIDADE:")
        report.append(f"   Estabilidade de retorno: {statistics['return_stability']:.1%}")
        report.append(f"   Coeficiente de variação: {statistics['coefficient_of_variation']:.3f}")

        # Recomendações baseadas nas estatísticas
        report.append(f"\n�� RECOMENDAÇÕES:")

        if statistics['probability_profit'] >= 0.7:
            report.append("   ✅ Alta probabilidade de lucro - estratégia robusta")
        elif statistics['probability_profit'] >= 0.6:
            report.append("   ✅ Boa probabilidade de lucro")
        else:
            report.append("   ⚠️ Probabilidade de lucro moderada - revisar estratégia")

        if statistics['var_95'] >= -0.15:
            report.append("   ✅ VaR aceitável - risco controlado")
        else:
            report.append("   ⚠️ VaR alto - considere reduzir exposição ao risco")

        if statistics['return_stability'] >= 0.6:
            report.append("   ✅ Retornos relativamente estáveis")
        else:
            report.append("   ⚠️ Retornos voláteis - estratégia pode ser inconsistente")

        if statistics['probability_drawdown_30pct'] <= 0.1:
            report.append("   ✅ Baixa probabilidade de drawdown severo")
        else:
            report.append("   ⚠️ Risco significativo de drawdown alto")

        return "\n".join(report)

    def confidence_intervals(self, mc_results: List[Dict[str, Any]],
                             confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calcula intervalos de confiança para as métricas"""

        df = pd.DataFrame(mc_results)

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        intervals = {}

        metrics = ['total_return', 'max_drawdown', 'sharpe_ratio', 'win_rate', 'profit_factor']

        for metric in metrics:
            if metric in df.columns:
                lower = df[metric].quantile(lower_percentile / 100)
                upper = df[metric].quantile(upper_percentile / 100)
                intervals[metric] = (lower, upper)

        return intervals