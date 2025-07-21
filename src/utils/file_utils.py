import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import os


class FileManager:
    """Gerenciador de arquivos para o sistema de trading"""

    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> bool:
        """Salva dados em formato JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar JSON {filepath}: {e}")
            return False

    @staticmethod
    def load_json(filepath: str) -> Optional[Dict[str, Any]]:
        """Carrega dados de arquivo JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Erro ao carregar JSON {filepath}: {e}")
            return None

    @staticmethod
    def save_pickle(data: Any, filepath: str) -> bool:
        """Salva dados em formato pickle"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar pickle {filepath}: {e}")
            return False

    @staticmethod
    def load_pickle(filepath: str) -> Optional[Any]:
        """Carrega dados de arquivo pickle"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Erro ao carregar pickle {filepath}: {e}")
            return None

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'csv') -> bool:
        """Salva DataFrame em diferentes formatos"""
        try:
            if format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'parquet':
                df.to_parquet(filepath, index=False)
            elif format == 'excel':
                df.to_excel(filepath, index=False)
            else:
                raise ValueError(f"Formato não suportado: {format}")
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar DataFrame {filepath}: {e}")
            return False

    @staticmethod
    def load_dataframe(filepath: str, format: str = None) -> Optional[pd.DataFrame]:
        """Carrega DataFrame detectando formato automaticamente"""
        try:
            filepath = Path(filepath)

            if format is None:
                format = filepath.suffix.lower()

            if format in ['.csv', 'csv']:
                return pd.read_csv(filepath)
            elif format in ['.parquet', 'parquet']:
                return pd.read_parquet(filepath)
            elif format in ['.xlsx', '.xls', 'excel']:
                return pd.read_excel(filepath)
            elif format in ['.json', 'json']:
                return pd.read_json(filepath)
            else:
                raise ValueError(f"Formato não suportado: {format}")

        except Exception as e:
            logging.error(f"Erro ao carregar DataFrame {filepath}: {e}")
            return None

    @staticmethod
    def ensure_directory(directory: str) -> bool:
        """Garante que o diretório existe"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logging.error(f"Erro ao criar diretório {directory}: {e}")
            return False

    @staticmethod
    def get_file_size(filepath: str) -> int:
        """Retorna tamanho do arquivo em bytes"""
        try:
            return Path(filepath).stat().st_size
        except Exception as e:
            logging.error(f"Erro ao obter tamanho do arquivo {filepath}: {e}")
            return 0

    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> list:
        """Lista arquivos em um diretório"""
        try:
            return list(Path(directory).glob(pattern))
        except Exception as e:
            logging.error(f"Erro ao listar arquivos em {directory}: {e}")
            return []

    @staticmethod
    def backup_file(filepath: str, backup_dir: str = "./backups") -> bool:
        """Cria backup de um arquivo"""
        try:
            import shutil
            from datetime import datetime

            filepath = Path(filepath)
            backup_dir = Path(backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            backup_path = backup_dir / backup_name

            shutil.copy2(filepath, backup_path)
            return True

        except Exception as e:
            logging.error(f"Erro ao fazer backup de {filepath}: {e}")
            return False

    @staticmethod
    def save_batch(batch, results_path, mode='a'):
        """Salva uma lista de dicionários/DataFrames em lote em CSV."""
        if not batch:
            return
        df = pd.DataFrame(batch)
        df.to_csv(results_path, mode=mode, header=not os.path.exists(results_path), index=False)