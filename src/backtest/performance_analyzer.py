import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging


class PerformanceAnalyzer:
    """Analisador de performance para backtests"""

    def __init__(self):
        self.setup_plotting_style()

    def setup_plotting_style(self):
        """Configura estilo dos gráficos"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def calculate_advanced_metrics(self, equity_curve: pd.DataFrame,
                                   trades_df: pd.DataFrame,
                                   benchmark_return: float = 0.1) -> Dict:
        """Calcula métricas avançadas de performance"""

        if equity_curve.empty:
            return {}

        # Preparar dados
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        equity_curve['cumulative_returns'] = (1 + equity_curve['returns']).cumprod() - 1

        # Métricas básicas
        total_return = equity_curve['cumulative_returns'].iloc[-1]
        annual_return = self._annualize_return(total_return, len(equity_curve))

        # Volatilidade
        daily_vol = equity_curve['returns'].std()
        annual_vol = daily_vol * np.sqrt(252)  # Assumindo dados diários

        # Sharpe Ratio
        risk_free_rate = 0.02  # 2% ao ano
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Sortino Ratio
        downside_returns = equity_curve['returns'][equity_curve['returns'] < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Calmar Ratio
        max_drawdown = self._calculate_max_drawdown(equity_curve['equity'])
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Information Ratio (vs benchmark)
        excess_returns = equity_curve['returns'] - (benchmark_return / 252)  # Daily benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

        # Beta (vs benchmark)
        benchmark_returns = np.full(len(equity_curve), benchmark_return / 252)
        beta = np.cov(equity_curve['returns'].dropna(), benchmark_returns[:len(equity_curve['returns'].dropna())])[
                   0, 1] / np.var(benchmark_returns)

        # Alpha
        alpha = annual_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

        # Métricas de trades
        if not trades_df.empty:
            win_rate = (trades_df['profit'] > 0).mean()
            profit_factor = trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(
                trades_df[trades_df['profit'] <= 0]['profit'].sum()) if (trades_df['profit'] <= 0).any() else float(
                'inf')
            avg_trade_return = trades_df['profit_pct'].mean() if 'profit_pct' in trades_df.columns else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_return = 0

        # Métricas de consistência
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        positive_months = (monthly_returns > 0).sum() if len(monthly_returns) > 0 else 0
        total_months = len(monthly_returns)
        consistency_ratio = positive_months / total_months if total_months > 0 else 0

        # Value at Risk
        var_95 = equity_curve['returns'].quantile(0.05)
        var_99 = equity_curve['returns'].quantile(0.01)

        # Expected Shortfall (CVaR)
        cvar_95 = equity_curve['returns'][equity_curve['returns'] <= var_95].mean()
        cvar_99 = equity_curve['returns'][equity_curve['returns'] <= var_99].mean()

        return {
            # Retornos
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return': annual_return / 12,

            # Risco
            'annual_volatility': annual_vol,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,

            # Ratios
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,

            # Benchmark comparison
            'alpha': alpha,
            'beta': beta,

            # Trading metrics
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,

            # Consistency
            'consistency_ratio': consistency_ratio,
            'positive_months': positive_months,
            'total_months': total_months
        }

    def _annualize_return(self, total_return: float, periods: int,
                          periods_per_year: int = 252) -> float:
        """Anualiza retorno"""
        years = periods / periods_per_year
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calcula drawdown máximo"""
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return drawdown.min()

    def _calculate_monthly_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """Calcula retornos mensais"""
        if 'timestamp' not in equity_curve.columns:
            return pd.Series()

        equity_curve = equity_curve.set_index('timestamp')
        monthly_equity = equity_curve['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()

        return monthly_returns

    def create_performance_report(self, results: Dict, save_path: str = None) -> str:
        """Cria relatório completo de performance"""

        report = []
        report.append("📊 RELATÓRIO DETALHADO DE PERFORMANCE")
        report.append("=" * 70)

        # Métricas principais
        metrics = results.get('advanced_metrics', {})

        if metrics:
            report.append(f"\n💰 RETORNOS:")
            report.append(f"   Retorno Total: {metrics.get('total_return', 0):.2%}")
            report.append(f"   Retorno Anualizado: {metrics.get('annual_return', 0):.2%}")
            report.append(f"   Retorno Mensal Médio: {metrics.get('monthly_return', 0):.2%}")

            report.append(f"\n📈 RISCO:")
            report.append(f"   Volatilidade Anual: {metrics.get('annual_volatility', 0):.2%}")
            report.append(f"   Drawdown Máximo: {metrics.get('max_drawdown', 0):.2%}")
            report.append(f"   VaR 95%: {metrics.get('var_95', 0):.2%}")
            report.append(f"   CVaR 95%: {metrics.get('cvar_95', 0):.2%}")

            report.append(f"\n⚡ RATIOS:")
            report.append(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            report.append(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
            report.append(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
            report.append(f"   Information Ratio: {metrics.get('information_ratio', 0):.3f}")

            report.append(f"\n🎯 TRADING:")
            report.append(f"   Taxa de Acerto: {metrics.get('win_rate', 0):.1%}")
            report.append(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            report.append(f"   Retorno Médio por Trade: {metrics.get('avg_trade_return', 0):.2%}")

            report.append(f"\n🔄 CONSISTÊNCIA:")
            report.append(f"   Meses Positivos: {metrics.get('positive_months', 0)}/{metrics.get('total_months', 0)}")
            report.append(f"   Ratio de Consistência: {metrics.get('consistency_ratio', 0):.1%}")

        # Classificação geral
        grade = self._calculate_strategy_grade(metrics)
        report.append(f"\n🏆 CLASSIFICAÇÃO GERAL: {grade}")

        # Recomendações
        recommendations = self._generate_recommendations(metrics)
        report.append(f"\n💡 RECOMENDAÇÕES:")
        for rec in recommendations:
            report.append(f"   {rec}")

        report_text = "\n".join(report)

        # Salvar se solicitado
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logging.info(f"Relatório salvo em: {save_path}")
            except Exception as e:
                logging.error(f"Erro ao salvar relatório: {e}")

        return report_text

    def _calculate_strategy_grade(self, metrics: Dict) -> str:
        """Calcula nota da estratégia"""

        if not metrics:
            return "N/A"

        score = 0

        # Retorno (0-25 pontos)
        annual_return = metrics.get('annual_return', 0)
        if annual_return >= 0.3:
            score += 25
        elif annual_return >= 0.2:
            score += 20
        elif annual_return >= 0.15:
            score += 15
        elif annual_return >= 0.1:
            score += 10
        elif annual_return >= 0.05:
            score += 5

        # Sharpe Ratio (0-20 pontos)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe >= 2.0:
            score += 20
        elif sharpe >= 1.5:
            score += 16
        elif sharpe >= 1.0:
            score += 12
        elif sharpe >= 0.5:
            score += 8
        elif sharpe >= 0:
            score += 4

        # Drawdown (0-20 pontos)
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd <= 0.05:
            score += 20
        elif max_dd <= 0.1:
            score += 16
        elif max_dd <= 0.15:
            score += 12
        elif max_dd <= 0.2:
            score += 8
        elif max_dd <= 0.3:
            score += 4

        # Win Rate (0-15 pontos)
        win_rate = metrics.get('win_rate', 0)
        if win_rate >= 0.7:
            score += 15
        elif win_rate >= 0.6:
            score += 12
        elif win_rate >= 0.5:
            score += 9
        elif win_rate >= 0.4:
            score += 6
        elif win_rate >= 0.3:
            score += 3

        # Consistência (0-10 pontos)
        consistency = metrics.get('consistency_ratio', 0)
        if consistency >= 0.8:
            score += 10
        elif consistency >= 0.7:
            score += 8
        elif consistency >= 0.6:
            score += 6
        elif consistency >= 0.5:
            score += 4
        elif consistency >= 0.4:
            score += 2

        # Profit Factor (0-10 pontos)
        pf = metrics.get('profit_factor', 0)
        if pf >= 3.0:
            score += 10
        elif pf >= 2.0:
            score += 8
        elif pf >= 1.5:
            score += 6
        elif pf >= 1.2:
            score += 4
        elif pf >= 1.0:
            score += 2

        # Classificar
        if score >= 90:
            return "A+ (Excepcional)"
        elif score >= 80:
            return "A (Excelente)"
        elif score >= 70:
            return "B+ (Muito Bom)"
        elif score >= 60:
            return "B (Bom)"
        elif score >= 50:
            return "C+ (Regular)"
        elif score >= 40:
            return "C (Abaixo da Média)"
        elif score >= 30:
            return "D+ (Ruim)"
        else:
            return "D (Muito Ruim)"

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Gera recomendações baseadas nas métricas"""

        recommendations = []

        if not metrics:
            return ["❌ Dados insuficientes para recomendações"]

        # Análise de retorno
        annual_return = metrics.get('annual_return', 0)
        if annual_return < 0.05:
            recommendations.append("📈 Retorno baixo - considere ajustar parâmetros para maior agressividade")
        elif annual_return > 0.5:
            recommendations.append("⚠️ Retorno muito alto - verifique se não há overfitting")

        # Análise de risco
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd > 0.2:
            recommendations.append("🛡️ Drawdown alto - implemente stop loss mais rigoroso")
        elif max_dd < 0.05:
            recommendations.append("✅ Drawdown controlado - boa gestão de risco")

        # Análise Sharpe
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < 0.5:
            recommendations.append("📊 Sharpe baixo - relação risco-retorno pode ser melhorada")
        elif sharpe > 1.5:
            recommendations.append("✅ Excelente Sharpe Ratio - boa relação risco-retorno")

        # Análise de win rate
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.4:
            recommendations.append("🎯 Win rate baixo - revise critérios de entrada")
        elif win_rate > 0.8:
            recommendations.append("⚠️ Win rate muito alto - pode indicar overfitting")

        # Análise de consistência
        consistency = metrics.get('consistency_ratio', 0)
        if consistency < 0.5:
            recommendations.append("🔄 Baixa consistência - estratégia pode ser instável")
        elif consistency > 0.7:
            recommendations.append("✅ Boa consistência temporal")

        # Análise profit factor
        pf = metrics.get('profit_factor', 0)
        if pf < 1.2:
            recommendations.append("💰 Profit factor baixo - lucros não superam perdas significativamente")
        elif pf > 3.0:
            recommendations.append("✅ Excelente profit factor")

        # Recomendações gerais
        if len(recommendations) == 0:
            recommendations.append("✅ Estratégia apresenta métricas equilibradas")

        return recommendations

    def create_visual_analysis(self, equity_curve: pd.DataFrame,
                               trades_df: pd.DataFrame,
                               save_dir: str = "./charts") -> Dict[str, str]:
        """Cria análise visual completa"""

        import os
        os.makedirs(save_dir, exist_ok=True)

        chart_files = {}

        try:
            # 1. Gráfico de equity curve
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(equity_curve['timestamp'], equity_curve['equity'], linewidth=2)
            ax.set_title('Curva de Capital', fontsize=16, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('Capital (R$)')
            ax.grid(True, alpha=0.3)

            equity_file = os.path.join(save_dir, 'equity_curve.png')
            plt.savefig(equity_file, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files['equity_curve'] = equity_file

            # 2. Gráfico de drawdown
            if not equity_curve.empty:
                equity_curve['peak'] = equity_curve['equity'].cummax()
                equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak']

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.fill_between(equity_curve['timestamp'], equity_curve['drawdown'], 0,
                                alpha=0.7, color='red')
                ax.set_title('Drawdown', fontsize=16, fontweight='bold')
                ax.set_xlabel('Data')
                ax.set_ylabel('Drawdown (%)')
                ax.grid(True, alpha=0.3)

                dd_file = os.path.join(save_dir, 'drawdown.png')
                plt.savefig(dd_file, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['drawdown'] = dd_file

            # 3. Distribuição de retornos
            if not trades_df.empty and 'profit_pct' in trades_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                trades_df['profit_pct'].hist(bins=30, alpha=0.7, ax=ax)
                ax.set_title('Distribuição de Retornos por Trade', fontsize=16, fontweight='bold')
                ax.set_xlabel('Retorno (%)')
                ax.set_ylabel('Frequência')
                ax.grid(True, alpha=0.3)

                returns_file = os.path.join(save_dir, 'returns_distribution.png')
                plt.savefig(returns_file, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files['returns_distribution'] = returns_file

            # 4. Performance mensal
            if not equity_curve.empty:
                monthly_returns = self._calculate_monthly_returns(equity_curve)

                if not monthly_returns.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    colors = ['green' if x > 0 else 'red' for x in monthly_returns]
                    monthly_returns.plot(kind='bar', color=colors, ax=ax)
                    ax.set_title('Retornos Mensais', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Mês')
                    ax.set_ylabel('Retorno (%)')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)

                    monthly_file = os.path.join(save_dir, 'monthly_returns.png')
                    plt.savefig(monthly_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files['monthly_returns'] = monthly_file

            logging.info(f"Gráficos salvos em: {save_dir}")

        except Exception as e:
            logging.error(f"Erro ao criar gráficos: {e}")

        return chart_files