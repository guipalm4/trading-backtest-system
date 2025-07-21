import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple


class DataValidator:
    """Validador e limpador de dados históricos"""

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Valida qualidade dos dados históricos"""

        issues = []

        if df is None or df.empty:
            return False, ["Dados vazios ou nulos"]

        # 1. Verificar colunas obrigatórias
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Colunas faltantes: {missing_columns}")

        # 2. Verificar dados faltantes
        missing_data = df[required_columns].isnull().sum()
        if missing_data.any():
            issues.append(f"Dados faltantes: {missing_data.to_dict()}")

        # 3. Verificar consistência OHLC
        invalid_ohlc = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
            ]
        if not invalid_ohlc.empty:
            issues.append(f"OHLC inconsistente em {len(invalid_ohlc)} registros")

        # 4. Verificar preços positivos
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            issues.append("Preços não positivos detectados")

        # 5. Verificar volume
        if (df['volume'] < 0).any():
            issues.append("Volume negativo detectado")

        # 6. Verificar duplicatas de timestamp
        duplicates = df.duplicated(subset=['timestamp']).sum()
        if duplicates > 0:
            issues.append(f"Timestamps duplicados: {duplicates}")

        # 7. Verificar ordem temporal
        if not df['timestamp'].is_monotonic_increasing:
            issues.append("Dados não estão em ordem cronológica")

        # 8. Verificar outliers extremos
        price_changes = df['close'].pct_change().abs()
        extreme_changes = (price_changes > 0.5).sum()  # Mudanças > 50%
        if extreme_changes > len(df) * 0.01:  # Mais de 1%
            issues.append(f"Muitos outliers extremos: {extreme_changes}")

        return len(issues) == 0, issues

    @staticmethod
    def clean_data(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """Limpa e trata dados históricos"""

        if df is None or df.empty:
            return df

        logging.info(f"🧹 Limpando dados: {len(df)} registros iniciais")
        original_length = len(df)

        # 1. Remover duplicatas
        df = df.drop_duplicates(subset=['timestamp'])
        if len(df) < original_length:
            logging.info(f"   Removidas {original_length - len(df)} duplicatas")

        # 2. Ordenar por timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 3. Remover registros com OHLC inválido
        invalid_mask = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close']) |
                (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        )

        if invalid_mask.any():
            df = df[~invalid_mask]
            logging.info(f"   Removidos {invalid_mask.sum()} registros com OHLC inválido")

        # 4. Tratar volume zero/negativo
        df.loc[df['volume'] <= 0, 'volume'] = df['volume'].median()

        # 5. Tratar outliers extremos
        if aggressive:
            price_changes = df['close'].pct_change().abs()
            outlier_mask = price_changes > 0.3  # Mudanças > 30%

            if outlier_mask.any():
                df = df[~outlier_mask]
                logging.info(f"   Removidos {outlier_mask.sum()} outliers extremos")

        # 6. Interpolar dados faltantes (limitado)
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method='linear', limit=3)

        # 7. Remover registros ainda com NaN
        df = df.dropna()

        # 8. Reset index
        df = df.reset_index(drop=True)

        logging.info(f"✅ Limpeza concluída: {len(df)} registros finais")

        return df

    @staticmethod
    def detect_gaps(df: pd.DataFrame, expected_interval_minutes: int) -> List[Dict]:
        """Detecta gaps temporais nos dados"""

        if df.empty:
            return []

        df['time_diff'] = df['timestamp'].diff()
        expected_diff = pd.Timedelta(minutes=expected_interval_minutes)

        # Gaps maiores que 2x o intervalo esperado
        large_gaps = df[df['time_diff'] > expected_diff * 2]

        gaps = []
        for idx, row in large_gaps.iterrows():
            if idx > 0:
                gaps.append({
                    'start_time': df.iloc[idx - 1]['timestamp'],
                    'end_time': row['timestamp'],
                    'duration': row['time_diff'],
                    'expected_duration': expected_diff,
                    'gap_ratio': row['time_diff'] / expected_diff
                })

        return gaps

    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> Dict:
        """Gera relatório de qualidade dos dados"""

        if df.empty:
            return {"error": "Dados vazios"}

        # Estatísticas básicas
        total_records = len(df)
        date_range = df['timestamp'].max() - df['timestamp'].min()

        # Verificar completude
        missing_data = df.isnull().sum()
        completeness = (1 - missing_data.sum() / (len(df) * len(df.columns))) * 100

        # Verificar consistência
        is_valid, issues = DataValidator.validate_data(df)

        # Estatísticas de preço
        price_stats = {
            'min_price': df['close'].min(),
            'max_price': df['close'].max(),
            'mean_price': df['close'].mean(),
            'price_volatility': df['close'].pct_change().std()
        }

        # Estatísticas de volume
        volume_stats = {
            'min_volume': df['volume'].min(),
            'max_volume': df['volume'].max(),
            'mean_volume': df['volume'].mean(),
            'zero_volume_count': (df['volume'] == 0).sum()
        }

        return {
            'total_records': total_records,
            'date_range_days': date_range.days,
            'start_date': df['timestamp'].min(),
            'end_date': df['timestamp'].max(),
            'completeness_pct': completeness,
            'is_valid': is_valid,
            'issues': issues,
            'price_stats': price_stats,
            'volume_stats': volume_stats
        }