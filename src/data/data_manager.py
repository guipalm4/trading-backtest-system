import os
import pickle
import hashlib
from pathlib import Path
import pandas as pd
import time
import logging
from typing import Optional, Dict, List
from .data_loader import DataLoader
from .data_validator import DataValidator


class DataManager:
    """Gerenciador de dados com cache inteligente"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_loader = DataLoader()

    def _generate_cache_key(self, symbol: str, interval: str, days_back: int, source: str) -> str:
        """Gera chave única para cache"""
        key_string = f"{symbol}_{interval}_{days_back}_{source}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Retorna caminho do arquivo de cache"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Verifica se cache ainda é válido"""
        if not cache_path.exists():
            return False

        file_age = time.time() - cache_path.stat().st_mtime
        max_age_seconds = max_age_hours * 3600

        return file_age < max_age_seconds

    def load_data(self, symbol: str, interval: str = '1h', days_back: int = 365,
                  source: str = 'binance', force_refresh: bool = False,
                  cache_hours: int = 24, clean_data: bool = True) -> Optional[pd.DataFrame]:
        """
        Carrega dados com cache inteligente e validação

        Args:
            symbol: Símbolo do ativo
            interval: Timeframe
            days_back: Dias para trás
            source: 'binance' ou 'yahoo'
            force_refresh: Forçar download
            cache_hours: Horas de validade do cache
            clean_data: Se deve limpar os dados
        """

        # Gerar chave de cache
        cache_key = self._generate_cache_key(symbol, interval, days_back, source)
        cache_path = self._get_cache_path(cache_key)

        # Verificar cache
        if not force_refresh and self._is_cache_valid(cache_path, cache_hours):
            try:
                logging.info(f"📁 Carregando do cache: {symbol}")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)

                logging.info(f"✅ Cache carregado: {len(data)} registros")
                return data

            except Exception as e:
                logging.warning(f"⚠️ Erro ao carregar cache: {e}")

        # Carregar dados da fonte
        logging.info(f"🌐 Baixando dados: {symbol} ({source})")

        if source == 'binance':
            data = self.data_loader.load_binance_data(symbol, interval, days_back)
        elif source == 'yahoo':
            data = self.data_loader.load_yahoo_data(symbol, interval, days_back)
        else:
            raise ValueError(f"Fonte não suportada: {source}")

        if data is None:
            return None

        # Validar e limpar dados
        if clean_data:
            is_valid, issues = DataValidator.validate_data(data)

            if issues:
                logging.warning(f"⚠️ Problemas nos dados: {issues}")
                data = DataValidator.clean_data(data, aggressive=False)

        # Salvar no cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"💾 Dados salvos no cache")
        except Exception as e:
            logging.warning(f"⚠️ Erro ao salvar cache: {e}")

        return data

    def clear_cache(self, symbol: str = None):
        """Limpa cache específico ou todo"""
        if symbol:
            for cache_file in self.cache_dir.glob(f"*{symbol}*.pkl"):
                cache_file.unlink()
                logging.info(f"🗑️ Cache removido: {cache_file.name}")
        else:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logging.info("🗑️ Todo o cache foi limpo")

    def get_cache_info(self) -> Dict:
        """Retorna informações sobre o cache"""
        cache_files = list(self.cache_dir.glob("*.pkl"))

        total_size = sum(f.stat().st_size for f in cache_files)

        files_info = []
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)

                file_age = time.time() - cache_file.stat().st_mtime

                files_info.append({
                    'file': cache_file.name,
                    'records': len(data),
                    'size_kb': cache_file.stat().st_size / 1024,
                    'age_hours': file_age / 3600,
                    'period': f"{data['timestamp'].min()} até {data['timestamp'].max()}"
                })

            except Exception as e:
                files_info.append({
                    'file': cache_file.name,
                    'error': str(e)
                })

        return {
            'total_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'files': files_info
        }

    def load_multiple_symbols(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Carrega dados de múltiplos símbolos"""

        datasets = {}

        logging.info(f"📊 Carregando {len(symbols)} símbolos")

        for i, symbol in enumerate(symbols, 1):
            logging.info(f"[{i}/{len(symbols)}] Processando {symbol}...")

            try:
                data = self.load_data(symbol, **kwargs)

                if data is not None and not data.empty:
                    datasets[symbol] = data
                    logging.info(f"✅ {symbol}: {len(data)} registros")
                else:
                    logging.error(f"❌ Falha ao carregar {symbol}")

            except Exception as e:
                logging.error(f"❌ Erro ao processar {symbol}: {e}")
                continue

        logging.info(f"📊 Carregamento concluído: {len(datasets)}/{len(symbols)} sucessos")
        return datasets

    def export_data(self, data: pd.DataFrame, symbol: str, format: str = 'csv') -> str:
        """Exporta dados para arquivo"""

        export_dir = self.cache_dir.parent / "exports"
        export_dir.mkdir(exist_ok=True)

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{timestamp}.{format}"
        filepath = export_dir / filename

        try:
            if format == 'csv':
                data.to_csv(filepath, index=False)
            elif format == 'parquet':
                data.to_parquet(filepath, index=False)
            elif format == 'json':
                data.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError(f"Formato não suportado: {format}")

            logging.info(f"💾 Dados exportados: {filepath}")
            return str(filepath)

        except Exception as e:
            logging.error(f"❌ Erro ao exportar: {e}")
            return ""