import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging


class Trade:
    """Classe para representar um trade individual"""

    def __init__(self, trade_id: int, symbol: str, side: str, quantity: float,
                 entry_price: float, entry_time: datetime):
        self.trade_id = trade_id
        self.symbol = symbol
        self.side = side  # 'buy' ou 'sell'
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time

        # Valores de saída (preenchidos quando o trade é fechado)
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.exit_reason: Optional[str] = None

        # Métricas calculadas
        self.profit: Optional[float] = None
        self.profit_pct: Optional[float] = None
        self.duration: Optional[timedelta] = None
        self.costs: Dict[str, float] = {}

        # Status
        self.is_open = True

    def close_trade(self, exit_price: float, exit_time: datetime,
                    exit_reason: str, costs: Dict[str, float]):
        """Fecha o trade e calcula métricas"""

        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.costs = costs
        self.is_open = False

        # Calcular lucro
        if self.side == 'buy':
            gross_profit = (exit_price - self.entry_price) * self.quantity
        else:  # sell (short)
            gross_profit = (self.entry_price - exit_price) * self.quantity

        total_costs = sum(costs.values())
        self.profit = gross_profit - total_costs
        self.profit_pct = self.profit / (self.entry_price * self.quantity)

        # Calcular duração
        self.duration = exit_time - self.entry_time

    def to_dict(self) -> Dict:
        """Converte trade para dicionário"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'exit_reason': self.exit_reason,
            'profit': self.profit,
            'profit_pct': self.profit_pct,
            'duration': self.duration,
            'costs': self.costs,
            'is_open': self.is_open
        }


class TradeManager:
    """Gerenciador de trades para backtesting"""

    def __init__(self):
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0

    def open_trade(self, symbol: str, side: str, quantity: float,
                   price: float, timestamp: datetime) -> Trade:
        """Abre um novo trade"""

        self.trade_counter += 1

        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp
        )

        self.trades.append(trade)
        self.open_trades.append(trade)

        logging.debug(f"Trade aberto: {trade.trade_id} - {side} {quantity} {symbol} @ {price}")

        return trade

    def close_trade(self, trade: Trade, exit_price: float, exit_time: datetime,
                    exit_reason: str, costs: Dict[str, float]) -> bool:
        """Fecha um trade existente"""

        if not trade.is_open:
            logging.warning(f"Tentativa de fechar trade já fechado: {trade.trade_id}")
            return False

        trade.close_trade(exit_price, exit_time, exit_reason, costs)

        # Mover para lista de trades fechados
        if trade in self.open_trades:
            self.open_trades.remove(trade)
        self.closed_trades.append(trade)

        logging.debug(f"Trade fechado: {trade.trade_id} - Lucro: {trade.profit:.2f} ({trade.profit_pct:.2%})")

        return True

    def close_all_trades(self, exit_price: float, exit_time: datetime,
                         exit_reason: str = "end_of_data", costs: Dict[str, float] = None):
        """Fecha todos os trades abertos"""

        if costs is None:
            costs = {'commission': 0, 'slippage': 0}

        for trade in self.open_trades.copy():
            self.close_trade(trade, exit_price, exit_time, exit_reason, costs)

    def get_open_positions(self, symbol: str = None) -> List[Trade]:
        """Retorna posições abertas"""

        if symbol:
            return [trade for trade in self.open_trades if trade.symbol == symbol]
        return self.open_trades.copy()

    def get_trade_statistics(self) -> Dict:
        """Calcula estatísticas dos trades"""

        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'avg_duration': timedelta(0)
            }

        closed_df = pd.DataFrame([trade.to_dict() for trade in self.closed_trades])

        # Estatísticas básicas
        total_trades = len(closed_df)
        winning_trades = len(closed_df[closed_df['profit'] > 0])
        losing_trades = len(closed_df[closed_df['profit'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Lucros e perdas
        total_profit = closed_df['profit'].sum()
        avg_profit = closed_df['profit'].mean()

        wins = closed_df[closed_df['profit'] > 0]['profit']
        losses = closed_df[closed_df['profit'] <= 0]['profit']

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        largest_win = wins.max() if len(wins) > 0 else 0
        largest_loss = losses.min() if len(losses) > 0 else 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Duração média
        avg_duration = closed_df['duration'].mean()

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_duration': avg_duration,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Retorna DataFrame com todos os trades"""

        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([trade.to_dict() for trade in self.trades])

    def get_monthly_performance(self) -> pd.DataFrame:
        """Calcula performance mensal"""

        if not self.closed_trades:
            return pd.DataFrame()

        df = self.get_trades_dataframe()
        df = df[df['is_open'] == False].copy()

        if df.empty:
            return pd.DataFrame()

        df['exit_month'] = pd.to_datetime(df['exit_time']).dt.to_period('M')

        monthly_stats = df.groupby('exit_month').agg({
            'profit': ['sum', 'count', 'mean'],
            'profit_pct': 'mean'
        }).round(4)

        monthly_stats.columns = ['total_profit', 'trades_count', 'avg_profit', 'avg_profit_pct']

        return monthly_stats

    def analyze_trade_patterns(self) -> Dict:
        """Analisa padrões nos trades"""

        if not self.closed_trades:
            return {}

        df = self.get_trades_dataframe()
        df = df[df['is_open'] == False].copy()

        if df.empty:
            return {}

        # Análise por hora do dia
        df['entry_hour'] = pd.to_datetime(df['entry_time']).dt.hour
        hourly_performance = df.groupby('entry_hour')['profit'].agg(['sum', 'count', 'mean'])

        # Análise por dia da semana
        df['entry_weekday'] = pd.to_datetime(df['entry_time']).dt.day_name()
        daily_performance = df.groupby('entry_weekday')['profit'].agg(['sum', 'count', 'mean'])

        # Análise por duração
        df['duration_hours'] = df['duration'].dt.total_seconds() / 3600
        duration_bins = pd.cut(df['duration_hours'], bins=[0, 1, 4, 12, 24, 72, float('inf')],
                               labels=['<1h', '1-4h', '4-12h', '12-24h', '24-72h', '>72h'])
        duration_performance = df.groupby(duration_bins)['profit'].agg(['sum', 'count', 'mean'])

        # Sequências de trades
        df['is_win'] = df['profit'] > 0
        df['win_streak'] = (df['is_win'] != df['is_win'].shift()).cumsum()

        win_streaks = df[df['is_win']].groupby('win_streak').size()
        loss_streaks = df[~df['is_win']].groupby('win_streak').size()

        return {
            'hourly_performance': hourly_performance.to_dict(),
            'daily_performance': daily_performance.to_dict(),
            'duration_performance': duration_performance.to_dict(),
            'max_win_streak': win_streaks.max() if len(win_streaks) > 0 else 0,
            'max_loss_streak': loss_streaks.max() if len(loss_streaks) > 0 else 0,
            'avg_win_streak': win_streaks.mean() if len(win_streaks) > 0 else 0,
            'avg_loss_streak': loss_streaks.mean() if len(loss_streaks) > 0 else 0
        }