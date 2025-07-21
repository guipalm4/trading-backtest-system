import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.data import DataManager
from src.data import DataValidator


class TestDataManager:
    """Testes para o DataManager"""

    def setup_method(self):
        """Setup para cada teste"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_manager = DataManager(cache_dir=self.temp_dir)

    def test_init(self):
        """Testa inicialização do DataManager"""
        assert self.data_manager.cache_dir.exists()
        assert isinstance(self.data_manager.validator, DataValidator)

    @patch('data.data_manager.yf.download')
    def test_load_data_yahoo_success(self, mock_download, sample_data):
        """Testa carregamento de dados do Yahoo Finance"""
        mock_download.return_value = sample_data.set_index('timestamp')

        result = self.data_manager.load_data('BTC-USD', '1h', 30, source='yahoo')

        assert result is not None
        assert len(result) > 0
        assert 'timestamp' in result.columns
        mock_download.assert_called_once()

    @patch('data.data_manager.yf.download')
    def test_load_data_yahoo_failure(self, mock_download):
        """Testa falha no carregamento do Yahoo Finance"""
        mock_download.side_effect = Exception("API Error")

        result = self.data_manager.load_data('INVALID', '1h', 30, source='yahoo')

        assert result is None

    @patch('data.data_manager.Client')
    def test_load_data_binance_success(self, mock_client_class, sample_data):
        """Testa carregamento de dados da Binance"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Simular resposta da Binance
        binance_data = []
        for _, row in sample_data.iterrows():
            binance_data.append([
                int(row['timestamp'].timestamp() * 1000),  # timestamp
                str(row['open']),
                str(row['high']),
                str(row['low']),
                str(row['close']),
                str(row['volume']),
                0, 0, 0, 0, 0, 0  # outros campos
            ])

        mock_client.get_historical_klines.return_value = binance_data

        result = self.data_manager.load_data('BTCBRL', '1h', 30, source='binance')

        assert result is not None
        assert len(result) > 0
        assert 'timestamp' in result.columns

    def test_cache_functionality(self, sample_data):
        """Testa funcionalidade de cache"""
        # Salvar dados no cache
        cache_file = self.data_manager._get_cache_filename('BTCBRL', '1h', 30)
        sample_data.to_parquet(cache_file, index=False)

        # Carregar do cache
        result = self.data_manager._load_from_cache('BTCBRL', '1h', 30)

        assert result is not None
        assert len(result) == len(sample_data)

    def test_cache_expiry(self, sample_data):
        """Testa expiração do cache"""
        # Criar cache antigo
        cache_file = self.data_manager._get_cache_filename('BTCBRL', '1h', 30)
        sample_data.to_parquet(cache_file, index=False)

        # Modificar timestamp para simular cache antigo
        old_time = os.path.getmtime(cache_file) - (25 * 3600)  # 25 horas atrás
        os.utime(cache_file, (old_time, old_time))

        result = self.data_manager._load_from_cache('BTCBRL', '1h', 30)

        assert result is None  # Cache expirado

    def test_validate_and_clean_data(self, sample_data):
        """Testa validação e limpeza de dados"""
        # Adicionar dados problemáticos
        dirty_data = sample_data.copy()
        dirty_data.loc[10, 'close'] = np.nan
        dirty_data.loc[20, 'volume'] = -100
        dirty_data.loc[30, 'high'] = dirty_data.loc[30, 'low'] - 1  # high < low

        cleaned_data = self.data_manager._validate_and_clean_data(dirty_data)

        assert cleaned_data is not None
        assert len(cleaned_data) <= len(dirty_data)  # Pode ter removido linhas
        assert not cleaned_data['close'].isna().any()
        assert (cleaned_data['volume'] >= 0).all()
        assert (cleaned_data['high'] >= cleaned_data['low']).all()

    def test_convert_interval(self):
        """Testa conversão de intervalos"""
        assert self.data_manager._convert_interval_to_binance('1m') == '1m'
        assert self.data_manager._convert_interval_to_binance('5m') == '5m'
        assert self.data_manager._convert_interval_to_binance('1h') == '1h'
        assert self.data_manager._convert_interval_to_binance('1d') == '1d'

    def test_load_multiple_symbols(self, sample_data):
        """Testa carregamento de múltiplos símbolos"""
        with patch.object(self.data_manager, 'load_data', return_value=sample_data):
            symbols = ['BTCBRL', 'ETHBRL']
            results = self.data_manager.load_multiple_symbols(symbols, interval='1h', days_back=30)

            assert len(results) == 2
            assert 'BTCBRL' in results
            assert 'ETHBRL' in results

    def test_export_data(self, sample_data):
        """Testa exportação de dados"""
        filepath = self.data_manager.export_data(sample_data, 'BTCBRL', 'csv')

        assert filepath != ""
        assert os.path.exists(filepath)

        # Verificar se arquivo foi criado corretamente
        exported_data = pd.read_csv(filepath)
        assert len(exported_data) == len(sample_data)


class TestDataValidator:
    """Testes para o DataValidator"""

    def setup_method(self):
        """Setup para cada teste"""
        self.validator = DataValidator()

    def test_validate_ohlcv_structure_valid(self, sample_data):
        """Testa validação de estrutura OHLCV válida"""
        is_valid, issues = self.validator.validate_ohlcv_structure(sample_data)

        assert is_valid
        assert len(issues) == 0

    def test_validate_ohlcv_structure_missing_columns(self, sample_data):
        """Testa validação com colunas faltando"""
        incomplete_data = sample_data.drop(columns=['volume'])

        is_valid, issues = self.validator.validate_ohlcv_structure(incomplete_data)

        assert not is_valid
        assert any('volume' in issue for issue in issues)

    def test_validate_ohlcv_structure_invalid_prices(self, sample_data):
        """Testa validação com preços inválidos"""
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'high'] = invalid_data.loc[0, 'low'] - 1  # high < low

        is_valid, issues = self.validator.validate_ohlcv_structure(invalid_data)

        assert not is_valid
        assert any('high < low' in issue for issue in issues)

    def test_detect_outliers(self, sample_data):
        """Testa detecção de outliers"""
        # Adicionar outlier óbvio
        outlier_data = sample_data.copy()
        outlier_data.loc[100, 'close'] = outlier_data['close'].mean() * 10

        outliers = self.validator.detect_outliers(outlier_data, 'close')

        assert len(outliers) > 0
        assert 100 in outliers

    def test_detect_missing_data(self, sample_data):
        """Testa detecção de dados faltando"""
        # Adicionar dados faltando
        missing_data = sample_data.copy()
        missing_data.loc[50:55, 'close'] = np.nan

        missing_info = self.validator.detect_missing_data(missing_data)

        assert missing_info['total_missing'] > 0
        assert 'close' in missing_info['missing_by_column']
        assert missing_info['missing_by_column']['close'] == 6

    def test_check_data_consistency(self, sample_data):
        """Testa verificação de consistência"""
        # Dados consistentes
        issues = self.validator.check_data_consistency(sample_data)
        assert len(issues) == 0

        # Adicionar inconsistência
        inconsistent_data = sample_data.copy()
        inconsistent_data.loc[10, 'high'] = inconsistent_data.loc[10, 'low'] - 1

        issues = self.validator.check_data_consistency(inconsistent_data)
        assert len(issues) > 0

    def test_clean_data(self, sample_data):
        """Testa limpeza de dados"""
        # Criar dados sujos
        dirty_data = sample_data.copy()
        dirty_data.loc[10, 'close'] = np.nan
        dirty_data.loc[20, 'volume'] = -100
        dirty_data.loc[30, 'close'] = dirty_data['close'].mean() * 10  # outlier

        cleaned_data = self.validator.clean_data(dirty_data)

        assert len(cleaned_data) <= len(dirty_data)
        assert not cleaned_data['close'].isna().any()
        assert (cleaned_data['volume'] >= 0).all()