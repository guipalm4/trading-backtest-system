import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from ..indicators.technical_indicators import TechnicalIndicators
from ..indicators.signal_generator import SignalGenerator


class BacktestEngine:
    """Engine principal de backtesting"""

    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()

    def reset(self):
        """Reset para novo backtest"""
        self.capital = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.trades = []
        self.equity_curve = []
        self.current_trade = None
        self.trade_count = 0

    def calculate_execution_cost(self, quantity: float, price: float, side: str = 'buy') -> Dict:
        """Calcula custos realistas de execução"""

        trade_value = quantity * price

        # Comissão
        commission_cost = trade_value * self.commission

        # Slippage (maior para vendas)
        slippage_multiplier = 1.5 if side == 'sell' else 1.0
        slippage_cost = trade_value * self.slippage * slippage_multiplier

        total_cost = commission_cost + slippage_cost

        return {
            'commission': commission_cost,
            'slippage': slippage_cost,
            'total': total_cost,
            'total_pct': total_cost / trade_value
        }

    def execute_buy(self, price: float, timestamp: pd.Timestamp, quantity: float = None) -> bool:
        """Executa ordem de compra"""

        if self.position > 0:  # Já tem posição
            return False

        # Calcular quantidade se não especificada
        if quantity is None:
            available_capital = self.capital * 0.95  # 5% de margem
            costs = self.calculate_execution_cost(1, price, 'buy')
            effective_price = price * (1 + costs['total_pct'])
            quantity = available_capital / effective_price

        if quantity <= 0:
            return False

        # Calcular custos
        costs = self.calculate_execution_cost(quantity, price, 'buy')
        total_cost = quantity * price + costs['total']

        if total_cost > self.capital:
            return False

        # Executar compra
        self.position = quantity
        self.position_value = quantity * price
        self.capital -= total_cost

        self.current_trade = {
            'type': 'buy',
            'price': price,
            'quantity': quantity,
            'timestamp': timestamp,
            'costs': costs,
            'trade_id': self.trade_count
        }

        self.trade_count += 1
        return True

    def execute_sell(self, price: float, timestamp: pd.Timestamp, reason: str = 'signal') -> bool:
        """Executa ordem de venda"""

        if self.position <= 0:  # Não tem posição
            return False

        # Calcular custos
        costs = self.calculate_execution_cost(self.position, price, 'sell')
        sell_value = self.position * price - costs['total']

        # Calcular lucro
        buy_costs = self.current_trade['costs']['total']
        total_profit = sell_value - self.position_value - buy_costs
        profit_pct = total_profit / self.position_value

        # Registrar trade completo
        trade_record = {
            'trade_id': self.current_trade['trade_id'],
            'buy_price': self.current_trade['price'],
            'sell_price': price,
            'quantity': self.position,
            'buy_timestamp': self.current_trade['timestamp'],
            'sell_timestamp': timestamp,
            'hold_time': timestamp - self.current_trade['timestamp'],
            'profit': total_profit,
            'profit_pct': profit_pct,
            'buy_costs': buy_costs,
            'sell_costs': costs['total'],
            'total_costs': buy_costs + costs['total'],
            'reason': reason
        }

        self.trades.append(trade_record)

        # Atualizar capital
        self.capital += sell_value

        # Reset posição
        self.position = 0
        self.position_value = 0
        self.current_trade = None

        return True

    def run_backtest(self, data: pd.DataFrame, config: Dict) -> Dict:
        """Executa backtest completo com configuração de estratégia parametrizável"""
        self.reset()
        # Calcular indicadores apenas os ativados
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(data, config)
        # Gerar sinais apenas com os ativados
        signal_generator = SignalGenerator(config)
        data_with_signals = signal_generator.generate_signals_for_backtest(data_with_indicators)
        # Simular trades
        for row in data_with_signals.itertuples(index=False):
            current_price = row.close
            timestamp = row.timestamp
            # Registrar equity
            current_equity = self.capital
            if self.position > 0:
                current_equity += self.position * current_price
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'price': current_price,
                'position': self.position > 0
            })
            # Executar sinais
            if row.buy_signal and self.position == 0:
                quantity = config.get('quantity', None)
                self.execute_buy(current_price, timestamp, quantity)
            elif row.sell_signal and self.position > 0:
                self.execute_sell(current_price, timestamp, row.signal_reason)
        # Fechar posição final se necessário
        if self.position > 0:
            final_row = data_with_signals.iloc[-1]
            self.execute_sell(final_row.close, final_row.timestamp, 'end_of_data')
        return self.calculate_performance_metrics()

    def calculate_performance_metrics(self) -> Dict:
        """Calcula métricas de performance"""

        if not self.trades:
            return self._get_empty_metrics()

        # Converter para DataFrames
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        # Métricas básicas
        final_capital = equity_df['equity'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # Estatísticas de trades
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Lucros e perdas
        gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(trades_df[trades_df['profit'] <= 0]['profit'].sum())
        net_profit = gross_profit - gross_loss

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy
        avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()

        # Sharpe ratio
        equity_df['returns'] = equity_df['equity'].pct_change()
        returns_mean = equity_df['returns'].mean()
        returns_std = equity_df['returns'].std()
        sharpe_ratio = (returns_mean / returns_std) * np.sqrt(252) if returns_std > 0 else 0

        # Outras métricas
        avg_trade_duration = trades_df['hold_time'].mean()
        total_costs = trades_df['total_costs'].sum()

        # Máxima sequência de perdas
        trades_df['is_loss'] = trades_df['profit'] <= 0
        max_consecutive_losses = self._calculate_max_consecutive(trades_df['is_loss'])

        # Máxima sequência de ganhos
        trades_df['is_win'] = trades_df['profit'] > 0
        max_consecutive_wins = self._calculate_max_consecutive(trades_df['is_win'])

        return {
            # Retornos
            'total_return': total_return,
            'net_profit': net_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,

            # Trades
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,

            # Métricas de risco-retorno
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,

            # Estatísticas de trades
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': trades_df['profit'].max(),
            'largest_loss': trades_df['profit'].min(),
            'avg_trade_duration': avg_trade_duration,

            # Custos
            'total_costs': total_costs,
            'avg_cost_per_trade': total_costs / total_trades if total_trades > 0 else 0,

            # Sequências
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,

            # Capital
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,

            # DataFrames para análise detalhada
            'equity_curve': equity_df,
            'trades_detail': trades_df
        }

    def _calculate_max_consecutive(self, boolean_series: pd.Series) -> int:
        """Calcula máxima sequência consecutiva de True"""
        if boolean_series.empty:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for value in boolean_series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _get_empty_metrics(self) -> Dict:
        """Retorna métricas vazias quando não há trades"""
        return {
            'total_return': 0,
            'net_profit': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'expectancy': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_duration': pd.Timedelta(0),
            'total_costs': 0,
            'avg_cost_per_trade': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'equity_curve': pd.DataFrame(),
            'trades_detail': pd.DataFrame()
        }