import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest.backtest_engine import BacktestEngine

class TestBacktestEngine:
    """Testes para o engine de backtest"""

    def setup_method(self):
        """Setup para cada teste"""
        self.engine = BacktestEngine(
            initial_capital=10000,
            commission=0.001,
            slippage=0.0005
        )

    def test_init(self):
        """Testa inicialização do engine"""
        assert self.engine.initial_capital == 10000
        assert self.engine.commission == 0.001
        assert self.engine.slippage == 0.0005
        assert self.engine.capital == 10000
        assert self.engine.position == 0

    def test_reset(self):
        """Testa reset do engine"""
        # Modificar estado
        self.engine.capital = 5000
        self.engine.position = 100
        self.engine.trades = [{'test': 'data'}]

        # Reset
        self.engine.reset()

        assert self.engine.capital == self.engine.initial_capital
        assert self.engine.position == 0
        assert len(self.engine.trades) == 0

    def test_calculate_execution_cost(self):
        """Testa cálculo de custos de execução"""
        quantity = 0.1
        price = 50000

        costs = self.engine.calculate_execution_cost(quantity, price, 'buy')

        assert 'commission' in costs
        assert 'slippage' in costs
        assert 'total' in costs
        assert 'total_pct' in costs

        assert costs['commission'] > 0
        assert costs['slippage'] > 0
        assert costs['total'] == costs['commission'] + costs['slippage']
        assert costs['total_pct'] > 0

    def test_execute_buy_success(self):
        """Testa execução de compra bem-sucedida"""
        price = 50000
        timestamp = datetime.now()

        success = self.engine.execute_buy(price, timestamp)

        assert success
        assert self.engine.position > 0
        assert self.engine.capital < self.engine.initial_capital
        assert self.engine.current_trade is not None

    def test_execute_buy_insufficient_capital(self):
        """Testa execução de compra com capital insuficiente"""
        # Usar preço muito alto para não ter capital suficiente
        price = 1000000
        timestamp = datetime.now()

        success = self.engine.execute_buy(price, timestamp)

        assert not success
        assert self.engine.position == 0
        assert self.engine.capital == self.engine.initial_capital

    def test_execute_buy_already_positioned(self):
        """Testa execução de compra quando já tem posição"""
        price = 50000
        timestamp = datetime.now()

        # Primeira compra
        self.engine.execute_buy(price, timestamp)
        initial_position = self.engine.position

        # Segunda compra (deve falhar)
        success = self.engine.execute_buy(price, timestamp)

        assert not success
        assert self.engine.position == initial_position

    def test_execute_sell_success(self):
        """Testa execução de venda bem-sucedida"""
        buy_price = 50000
        sell_price = 51000
        timestamp = datetime.now()

        # Primeiro comprar
        self.engine.execute_buy(buy_price, timestamp)
        initial_trades = len(self.engine.trades)

        # Depois vender
        success = self.engine.execute_sell(sell_price, timestamp + timedelta(hours=1))

        assert success
        assert self.engine.position == 0
        assert len(self.engine.trades) == initial_trades + 1
        assert self.engine.capital > self.engine.initial_capital  # Lucro

    def test_execute_sell_no_position(self):
        """Testa execução de venda sem posição"""
        price = 50000
        timestamp = datetime.now()

        success = self.engine.execute_sell(price, timestamp)

        assert not success
        assert len(self.engine.trades) == 0

    def test_run_backtest_complete(self, sample_data, sample_config):
        """Testa execução completa de backtest"""
        results = self.engine.run_backtest(sample_data, sample_config)

        # Verificar estrutura dos resultados
        expected_keys = [
            'total_return', 'net_profit', 'gross_profit', 'gross_loss',
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'profit_factor', 'expectancy', 'max_drawdown', 'sharpe_ratio',
            'avg_win', 'avg_loss', 'largest_win', 'largest_loss',
            'avg_trade_duration', 'total_costs', 'initial_capital',
            'final_capital', 'equity_curve', 'trades_detail'
        ]

        for key in expected_keys:
            assert key in results

        # Verificar tipos e valores
        assert isinstance(results['total_return'], float)
        assert isinstance(results['total_trades'], int)
        assert isinstance(results['win_rate'], float)
        assert isinstance(results['equity_curve'], pd.DataFrame)
        assert isinstance(results['trades_detail'], pd.DataFrame)

        # Verificar consistência
        assert results['total_trades'] == results['winning_trades'] + results['losing_trades']
        assert 0 <= results['win_rate'] <= 1
        assert results['final_capital'] == results['initial_capital'] + results['net_profit']

    def test_calculate_metrics(self):
        """Testa cálculo de métricas"""
        # Simular alguns trades
        self.engine.trades = [
            {'profit': 100, 'profit_pct': 0.02, 'duration': timedelta(hours=2)},
            {'profit': -50, 'profit_pct': -0.01, 'duration': timedelta(hours=1)},
            {'profit': 200, 'profit_pct': 0.04, 'duration': timedelta(hours=3)},
            {'profit': -25, 'profit_pct': -0.005, 'duration': timedelta(hours=1.5)}
        ]

        # Simular equity curve
        equity_curve = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'equity': np.random.uniform(9000, 11000, 100)
        })

        metrics = self.engine.calculate_metrics(equity_curve)

        assert metrics['total_trades'] == 4
        assert metrics['winning_trades'] == 2
        assert metrics['losing_trades'] == 2
        assert metrics['win_rate'] == 0.5
        assert metrics['gross_profit'] == 300
        assert metrics['gross_loss'] == 75
        assert metrics['profit_factor'] == 4.0

    def test_calculate_sharpe_ratio(self):
        """Testa cálculo do Sharpe Ratio"""
        # Equity curve com retornos positivos
        equity_curve = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'equity': np.linspace(10000, 12000, 100)  # Crescimento linear
        })

        sharpe = self.engine.calculate_sharpe_ratio(equity_curve)

        assert isinstance(sharpe, float)
        assert sharpe > 0  # Deve ser positivo para retornos crescentes

    def test_calculate_max_drawdown(self):
        """Testa cálculo do drawdown máximo"""
        # Equity curve com drawdown conhecido
        equity_values = [10000, 11000, 10500, 9000, 9500, 12000]  # DD de 18.18%
        equity_curve = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=6, freq='1H'),
            'equity': equity_values
        })

        max_dd = self.engine.calculate_max_drawdown(equity_curve)

        assert isinstance(max_dd, float)
        assert max_dd < 0  # Drawdown é negativo
        assert abs(max_dd) > 0.15  # Deve detectar o drawdown de ~18%

    def test_backtest_with_no_signals(self, sample_data, sample_config):
        """Testa backtest sem sinais de compra/venda"""
        # Configurar para não gerar sinais
        no_signal_config = sample_config.copy()
        no_signal_config['min_score'] = 100  # Score impossível

        results = self.engine.run_backtest(sample_data, no_signal_config)

        assert results['total_trades'] == 0
        assert results['total_return'] == 0
        assert results['final_capital'] == results['initial_capital']

    def test_backtest_with_costs(self, sample_data, sample_config):
        """Testa backtest considerando custos"""
        # Engine com custos altos
        expensive_engine = BacktestEngine(
            initial_capital=10000,
            commission=0.01,  # 1%
            slippage=0.005  # 0.5%
        )

        results = expensive_engine.run_backtest(sample_data, sample_config)

        if results['total_trades'] > 0:
            assert results['total_costs'] > 0
            # Com custos altos, retorno deve ser menor
            cheap_results = self.engine.run_backtest(sample_data, sample_config)
            if cheap_results['total_trades'] > 0:
                assert results['total_return'] <= cheap_results['total_return']

    class TestTradeExecution:
        """Testes específicos para execução de trades"""

        def setup_method(self):
            """Setup para cada teste"""
            self.engine = BacktestEngine(initial_capital=10000)

        def test_trade_lifecycle(self):
            """Testa ciclo completo de um trade"""
            buy_price = 50000
            sell_price = 52000
            buy_time = datetime(2023, 1, 1, 10, 0)
            sell_time = datetime(2023, 1, 1, 12, 0)

            # Compra
            buy_success = self.engine.execute_buy(buy_price, buy_time)
            assert buy_success
            assert self.engine.current_trade is not None
            assert self.engine.position > 0

            # Venda
            sell_success = self.engine.execute_sell(sell_price, sell_time, 'take_profit')
            assert sell_success
            assert self.engine.current_trade is None
            assert self.engine.position == 0
            assert len(self.engine.trades) == 1

            # Verificar trade registrado
            trade = self.engine.trades[0]
            assert trade['entry_price'] == buy_price
            assert trade['exit_price'] == sell_price
            assert trade['entry_time'] == buy_time
            assert trade['exit_time'] == sell_time
            assert trade['exit_reason'] == 'take_profit'
            assert trade['profit'] > 0  # Deve ter lucro

        def test_multiple_trades(self):
            """Testa múltiplos trades sequenciais"""
            trades_data = [
                (50000, 51000, 'take_profit'),
                (49000, 48000, 'stop_loss'),
                (47000, 49000, 'technical_signal')
            ]

            for i, (buy_price, sell_price, reason) in enumerate(trades_data):
                buy_time = datetime(2023, 1, 1, 10 + i * 2, 0)
                sell_time = datetime(2023, 1, 1, 11 + i * 2, 0)

                self.engine.execute_buy(buy_price, buy_time)
                self.engine.execute_sell(sell_price, sell_time, reason)

            assert len(self.engine.trades) == 3

            # Verificar que trades foram registrados corretamente
            for i, trade in enumerate(self.engine.trades):
                expected_buy, expected_sell, expected_reason = trades_data[i]
                assert trade['entry_price'] == expected_buy
                assert trade['exit_price'] == expected_sell
                assert trade['exit_reason'] == expected_reason

        def test_position_sizing(self):
            """Testa cálculo do tamanho da posição"""
            price = 50000
            timestamp = datetime.now()

            initial_capital = self.engine.capital
            self.engine.execute_buy(price, timestamp)

            # Verificar que posição foi calculada corretamente
            expected_quantity = (initial_capital * 0.95) / price  # 95% do capital
            assert abs(self.engine.position - expected_quantity) < 0.0001