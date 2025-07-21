import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict
import yfinance as yf


class DataLoader:
    """Carregador de dados históricos"""

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.client = Client(api_key, api_secret) if api_key else Client()

    def load_binance_data(self, symbol: str, interval: str, days_back: int = 365) -> Optional[pd.DataFrame]:
        """Carrega dados históricos da Binance"""

        try:
            logging.info(f"📊 Carregando dados Binance: {symbol} - {interval} - {days_back} dias")

            # Calcular data de início
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)

            # Carregar dados em lotes
            all_klines = []
            current_start = start_time

            while current_start < end_time:
                # Calcular tamanho do lote baseado no intervalo
                if interval in [Client.KLINE_INTERVAL_1MINUTE, Client.KLINE_INTERVAL_5MINUTE]:
                    batch_days = 1
                elif interval in [Client.KLINE_INTERVAL_15MINUTE, Client.KLINE_INTERVAL_30MINUTE]:
                    batch_days = 7
                elif interval == Client.KLINE_INTERVAL_1HOUR:
                    batch_days = 30
                else:
                    batch_days = days_back

                batch_end = min(current_start + timedelta(days=batch_days), end_time)

                logging.info(f"   Lote: {current_start.strftime('%Y-%m-%d')} até {batch_end.strftime('%Y-%m-%d')}")

                klines = self.client.get_historical_klines(
                    symbol,
                    interval,
                    start_str=current_start.strftime('%Y-%m-%d'),
                    end_str=batch_end.strftime('%Y-%m-%d')
                )

                all_klines.extend(klines)
                current_start = batch_end
                time.sleep(0.1)  # Rate limiting

            if not all_klines:
                logging.error(f"Nenhum dado encontrado para {symbol}")
                return None

            # Converter para DataFrame
            df = pd.DataFrame(all_klines)
            df.columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]

            # Converter tipos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Converter timestamps
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert('America/Sao_Paulo')

            # Selecionar colunas necessárias
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df = df.dropna().drop_duplicates(subset=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            logging.info(f"✅ Dados carregados: {len(df)} registros")
            return df

        except Exception as e:
            logging.error(f"❌ Erro ao carregar dados Binance: {e}")
            return None

    def load_yahoo_data(self, symbol: str, interval: str = '1h', days_back: int = 365) -> Optional[pd.DataFrame]:
        """Carrega dados do Yahoo Finance"""

        try:
            logging.info(f"📊 Carregando dados Yahoo: {symbol}")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval,
                auto_adjust=True
            )

            if data.empty:
                logging.error(f"Nenhum dado encontrado para {symbol}")
                return None

            # Padronizar colunas
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]

            column_mapping = {
                'datetime': 'timestamp',
                'date': 'timestamp'
            }
            data = data.rename(columns=column_mapping)

            # Verificar colunas necessárias
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                logging.error(f"Colunas faltantes nos dados do Yahoo")
                return None

            data = data[required_columns].copy()
            data = data.dropna().drop_duplicates(subset=['timestamp'])
            data = data.sort_values('timestamp').reset_index(drop=True)

            logging.info(f"✅ Dados Yahoo carregados: {len(data)} registros")
            return data

        except Exception as e:
            logging.error(f"❌ Erro ao carregar dados Yahoo: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Dict:
        """Obtém informações do símbolo"""

        try:
            symbol_info = self.client.get_symbol_info(symbol)

            # Extrair informações relevantes
            lot_size_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
            price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')

            return {
                'symbol': symbol,
                'status': symbol_info['status'],
                'base_asset': symbol_info['baseAsset'],
                'quote_asset': symbol_info['quoteAsset'],
                'min_qty': float(lot_size_filter['minQty']),
                'max_qty': float(lot_size_filter['maxQty']),
                'step_size': float(lot_size_filter['stepSize']),
                'min_price': float(price_filter['minPrice']),
                'max_price': float(price_filter['maxPrice']),
                'tick_size': float(price_filter['tickSize'])
            }

        except Exception as e:
            logging.error(f"❌ Erro ao obter info do símbolo: {e}")
            return {}