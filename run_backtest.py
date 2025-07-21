#!/usr/bin/env python3
"""
Script principal para executar backtests e otimização de parâmetros
"""

import logging
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

from config.settings import STRATEGY_CONFIG
from config.parameters import ParameterSpace
from src.data import DataManager
from src.backtest.backtest_engine import BacktestEngine
from src.optimization.parameter_optimizer import ParameterOptimizer
from src.optimization.walk_forward import WalkForwardAnalyzer
from src.optimization.monte_carlo import MonteCarloValidator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)


def run_single_backtest():
    """Executa um backtest simples com parâmetros padrão"""

    print("🚀 EXECUTANDO BACKTEST SIMPLES")
    print("=" * 50)

    # Carregar dados
    data_manager = DataManager()
    data = data_manager.load_data(
        symbol=STRATEGY_CONFIG.operation_code,
        interval='1h',
        days_back=STRATEGY_CONFIG.days_back,
        source='binance'
    )

    if data is None:
        print("❌ Erro ao carregar dados")
        return

    # Configurar parâmetros
    config = STRATEGY_CONFIG.__dict__.copy()

    # Executar backtest
    engine = BacktestEngine(
        initial_capital=STRATEGY_CONFIG.initial_capital,
        commission=STRATEGY_CONFIG.commission,
        slippage=STRATEGY_CONFIG.slippage
    )

    results = engine.run_backtest(data, config)

    # Exibir resultados
    print_backtest_results(results, "BACKTEST SIMPLES")

    # Salvar resultados
    save_results(results, "single_backtest", config)

    return results


def run_parameter_optimization():
    """Executa otimização completa de parâmetros"""

    print("🔬 EXECUTANDO OTIMIZAÇÃO DE PARÂMETROS")
    print("=" * 50)

    # Escolher perfil de estratégia
    print("\nEscolha o perfil de estratégia para otimização:")
    print("1. Scalping (trades rápidos, stops curtos)")
    print("2. Swing (operações mais longas)")
    print("3. Default (genérico)")
    perfil = input("Digite sua escolha (1-3): ").strip()
    if perfil == "1":
        parameter_space = ParameterSpace.get_scalping_space()
        print("\nPerfil selecionado: Scalping")
        score_weights = {
            'total_return': 0.10,
            'max_drawdown': 0.40,
            'sharpe_ratio': 0.10,
            'win_rate': 0.20,
            'profit_factor': 0.20
        }
    elif perfil == "2":
        parameter_space = ParameterSpace.get_swing_space()
        print("\nPerfil selecionado: Swing")
        score_weights = {
            'total_return': 0.30,
            'max_drawdown': 0.20,
            'sharpe_ratio': 0.20,
            'win_rate': 0.15,
            'profit_factor': 0.15
        }
    else:
        parameter_space = ParameterSpace.get_default_space()
        print("\nPerfil selecionado: Default")
        score_weights = None  # Usar padrão do otimizador

    # Carregar dados
    data_manager = DataManager()
    data = data_manager.load_data(
        symbol=STRATEGY_CONFIG.operation_code,
        interval='1h',
        days_back=STRATEGY_CONFIG.days_back,
        source='binance'
    )

    if data is None:
        print("❌ Erro ao carregar dados")
        return

    # Dividir dados para validação
    train_size = int(len(data) * STRATEGY_CONFIG.train_ratio)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()

    print(f"📊 Dados de treino: {len(train_data)} registros")
    print(f"📊 Dados de teste: {len(test_data)} registros")

    # Otimizar parâmetros
    optimizer = ParameterOptimizer()

    best_params, optimization_results = optimizer.optimize_parameters(
        data=train_data,
        parameter_space=parameter_space,
        max_combinations=200,
        n_jobs=4,  # Processamento paralelo
        score_weights=score_weights
    )

    print(f"\n🏆 MELHORES PARÂMETROS ENCONTRADOS:")
    print("-" * 40)
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Validar no conjunto de teste
    print(f"\n🧪 VALIDANDO NO CONJUNTO DE TESTE...")

    # Mesclar configuração padrão com parâmetros otimizados
    final_config = STRATEGY_CONFIG.__dict__.copy()
    final_config.update(best_params)

    engine = BacktestEngine(
        initial_capital=STRATEGY_CONFIG.initial_capital,
        commission=STRATEGY_CONFIG.commission
    )

    test_results = engine.run_backtest(test_data, final_config)

    print_backtest_results(test_results, "VALIDAÇÃO OUT-OF-SAMPLE")

    # Salvar resultados completos
    save_optimization_results(optimization_results, best_params, test_results)

    return best_params, optimization_results, test_results


def run_walk_forward_analysis():
    """Executa análise walk-forward"""

    print("📈 EXECUTANDO ANÁLISE WALK-FORWARD")
    print("=" * 50)

    # Carregar dados
    data_manager = DataManager()
    data = data_manager.load_data(
        symbol=STRATEGY_CONFIG.operation_code,
        interval='1h',
        days_back=STRATEGY_CONFIG.days_back,
        source='binance'
    )

    if data is None:
        print("❌ Erro ao carregar dados")
        return

    # Executar walk-forward
    wf_analyzer = WalkForwardAnalyzer()

    parameter_space = ParameterSpace.get_default_space()

    wf_results = wf_analyzer.run_walk_forward_analysis(
        data=data,
        parameter_space=parameter_space,
        n_windows=STRATEGY_CONFIG.walk_forward_windows,
        optimization_length_pct=0.7,
        max_combinations_per_window=100
    )

    # Analisar consistência
    consistency_metrics = wf_analyzer.analyze_consistency(wf_results)

    print(f"\n📊 ANÁLISE DE CONSISTÊNCIA:")
    print("-" * 40)
    print(f"Retorno médio: {consistency_metrics['avg_return']:.2%}")
    print(f"Desvio padrão: {consistency_metrics['std_return']:.2%}")
    print(f"Períodos positivos: {consistency_metrics['positive_periods']}/{consistency_metrics['total_periods']}")
    print(f"Sharpe médio: {consistency_metrics['avg_sharpe']:.3f}")
    print(f"Drawdown médio: {consistency_metrics['avg_drawdown']:.2%}")

    # Salvar resultados
    save_walk_forward_results(wf_results, consistency_metrics)

    return wf_results, consistency_metrics


def run_monte_carlo_validation():
    """Executa validação Monte Carlo"""

    print("🎲 EXECUTANDO VALIDAÇÃO MONTE CARLO")
    print("=" * 50)

    # Carregar dados
    data_manager = DataManager()
    data = data_manager.load_data(
        symbol=STRATEGY_CONFIG.operation_code,
        interval='1h',
        days_back=STRATEGY_CONFIG.days_back,
        source='binance'
    )

    if data is None:
        print("❌ Erro ao carregar dados")
        return

    # Usar parâmetros padrão ou otimizados
    config = STRATEGY_CONFIG.__dict__.copy()

    # Executar Monte Carlo
    mc_validator = MonteCarloValidator()

    mc_results = mc_validator.run_monte_carlo_simulation(
        data=data,
        config=config,
        n_simulations=STRATEGY_CONFIG.monte_carlo_simulations,
        sample_ratio=0.8
    )

    # Analisar estatísticas
    statistics = mc_validator.calculate_statistics(mc_results)

    print(f"\n📊 ESTATÍSTICAS MONTE CARLO:")
    print("-" * 40)
    print(f"Retorno médio: {statistics['mean_return']:.2%}")
    print(f"Desvio padrão: {statistics['std_return']:.2%}")
    print(f"VaR 95%: {statistics['var_95']:.2%}")
    print(f"CVaR 95%: {statistics['cvar_95']:.2%}")
    print(f"Probabilidade de lucro: {statistics['probability_profit']:.1%}")
    print(f"Probabilidade de perda > 10%: {statistics['probability_loss_10pct']:.1%}")

    # Salvar resultados
    save_monte_carlo_results(mc_results, statistics)

    return mc_results, statistics


def print_backtest_results(results: dict, title: str):
    """Imprime resultados do backtest de forma organizada"""

    print(f"\n📊 {title}")
    print("=" * 60)

    # Resultados financeiros
    print(f"💰 RESULTADOS FINANCEIROS:")
    print(f"   Capital inicial: R$ {results['initial_capital']:,.2f}")
    print(f"   Capital final: R$ {results['final_capital']:,.2f}")
    print(f"   Retorno total: {results['total_return']:.2%}")
    print(f"   Lucro líquido: R$ {results['net_profit']:,.2f}")

    # Estatísticas de trades
    print(f"\n🔢 ESTATÍSTICAS DE TRADES:")
    print(f"   Total de trades: {results['total_trades']}")
    print(f"   Trades vencedores: {results['winning_trades']}")
    print(f"   Trades perdedores: {results['losing_trades']}")
    print(f"   Taxa de acerto: {results['win_rate']:.1%}")

    # Métricas de risco-retorno
    print(f"\n📈 MÉTRICAS DE RISCO-RETORNO:")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Expectancy: R$ {results['expectancy']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.3f}")

    # Estatísticas de trades
    if results['total_trades'] > 0:
        print(f"\n💹 ESTATÍSTICAS DETALHADAS:")
        print(f"   Lucro médio: R$ {results['avg_win']:.2f}")
        print(f"   Perda média: R$ {results['avg_loss']:.2f}")
        print(f"   Maior lucro: R$ {results['largest_win']:.2f}")
        print(f"   Maior perda: R$ {results['largest_loss']:.2f}")
        print(f"   Duração média: {results['avg_trade_duration']}")
        print(f"   Custos totais: R$ {results['total_costs']:.2f}")


def save_results(results: dict, test_name: str, config: dict):
    """Salva resultados em arquivo"""
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.json"

    # Preparar dados para salvar (converter DataFrames)
    save_data = {
        'timestamp': timestamp,
        'test_name': test_name,
        'config': config,
        'results': {k: v for k, v in results.items() if k not in ['equity_curve', 'trades_detail']},
        'summary': {
            'total_return': results['total_return'],
            'max_drawdown': results['max_drawdown'],
            'sharpe_ratio': results['sharpe_ratio'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate']
        }
    }

    # Salvar JSON
    with open(results_dir / filename, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    # Salvar DataFrames separadamente
    if 'equity_curve' in results and not results['equity_curve'].empty:
        results['equity_curve'].to_csv(results_dir / f"equity_curve_{timestamp}.csv", index=False)

    if 'trades_detail' in results and not results['trades_detail'].empty:
        results['trades_detail'].to_csv(results_dir / f"trades_detail_{timestamp}.csv", index=False)

    print(f"💾 Resultados salvos: {filename}")


def save_optimization_results(optimization_results: list, best_params: dict, test_results: dict):
    """Salva resultados da otimização"""
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Salvar resultados da otimização
    opt_df = pd.DataFrame(optimization_results)
    opt_df.to_csv(results_dir / f"optimization_results_{timestamp}.csv", index=False)

    # Salvar melhores parâmetros
    with open(results_dir / f"best_parameters_{timestamp}.json", 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'best_parameters': best_params,
            'test_results_summary': {
                'total_return': test_results['total_return'],
                'max_drawdown': test_results['max_drawdown'],
                'sharpe_ratio': test_results['sharpe_ratio'],
                'total_trades': test_results['total_trades'],
                'win_rate': test_results['win_rate']
            }
        }, f, indent=2, default=str)

    print(f"💾 Resultados de otimização salvos: optimization_results_{timestamp}.csv")


def save_walk_forward_results(wf_results: list, consistency_metrics: dict):
    """Salva resultados walk-forward"""
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Salvar resultados
    wf_df = pd.DataFrame(wf_results)
    wf_df.to_csv(results_dir / f"walk_forward_results_{timestamp}.csv", index=False)

    # Salvar métricas de consistência
    with open(results_dir / f"walk_forward_consistency_{timestamp}.json", 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'consistency_metrics': consistency_metrics
        }, f, indent=2, default=str)

    print(f"💾 Resultados walk-forward salvos: walk_forward_results_{timestamp}.csv")


def save_monte_carlo_results(mc_results: list, statistics: dict):
    """Salva resultados Monte Carlo"""
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Salvar resultados
    mc_df = pd.DataFrame(mc_results)
    mc_df.to_csv(results_dir / f"monte_carlo_results_{timestamp}.csv", index=False)

    # Salvar estatísticas
    with open(results_dir / f"monte_carlo_statistics_{timestamp}.json", 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'statistics': statistics
        }, f, indent=2, default=str)

    print(f"💾 Resultados Monte Carlo salvos: monte_carlo_results_{timestamp}.csv")


def main():
    """Função principal"""

    print("🚀 SISTEMA DE BACKTESTING E OTIMIZAÇÃO")
    print("=" * 60)
    print("Escolha uma opção:")
    print("1. Backtest simples")
    print("2. Otimização de parâmetros")
    print("3. Análise walk-forward")
    print("4. Validação Monte Carlo")
    print("5. Análise completa (todos os testes)")
    print("0. Sair")

    choice = input("\nDigite sua escolha (0-5): ").strip()

    if choice == "1":
        run_single_backtest()

    elif choice == "2":
        run_parameter_optimization()

    elif choice == "3":
        run_walk_forward_analysis()

    elif choice == "4":
        run_monte_carlo_validation()

    elif choice == "5":
        print("\n🔬 EXECUTANDO ANÁLISE COMPLETA")
        print("=" * 50)

        # 1. Backtest simples
        print("\n1️⃣ Executando backtest simples...")
        simple_results = run_single_backtest()

        # 2. Otimização
        print("\n2️⃣ Executando otimização...")
        best_params, opt_results, test_results = run_parameter_optimization()

        # 3. Walk-forward com melhores parâmetros
        print("\n3️⃣ Executando walk-forward...")
        wf_results, consistency = run_walk_forward_analysis()

        # 4. Monte Carlo com melhores parâmetros
        print("\n4️⃣ Executando Monte Carlo...")
        mc_results, mc_stats = run_monte_carlo_validation()

        # Resumo final
        print("\n🎯 RESUMO FINAL DA ANÁLISE COMPLETA")
        print("=" * 60)
        print(f"✅ Backtest simples: {simple_results['total_return']:.2%} retorno")
        print(f"✅ Melhor configuração: {test_results['total_return']:.2%} retorno (out-of-sample)")
        print(
            f"✅ Consistência walk-forward: {consistency['positive_periods']}/{consistency['total_periods']} períodos positivos")
        print(f"✅ Monte Carlo: {mc_stats['probability_profit']:.1%} probabilidade de lucro")

        print(f"\n🏆 RECOMENDAÇÃO FINAL:")
        if (test_results['total_return'] > 0.1 and
                consistency['positive_periods'] / consistency['total_periods'] >= 0.6 and
                mc_stats['probability_profit'] >= 0.6):
            print("✅ Estratégia APROVADA para trading ao vivo")
        else:
            print("⚠️ Estratégia precisa de mais ajustes antes do trading ao vivo")

    elif choice == "0":
        print("👋 Saindo...")
        return

    else:
        print("❌ Opção inválida")
        main()


if __name__ == "__main__":
    main()