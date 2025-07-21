# Trading Backtest System

Sistema profissional de backtesting e otimização de estratégias de trading.

---

## Principais Features

- **Configuração centralizada e extensível**: Todos os parâmetros de trading, indicadores e sinais são definidos em `StrategyConfig` (`config/settings.py`).
- **Indicadores parametrizáveis**: Ative/desative indicadores e ajuste seus parâmetros facilmente.
- **Espaço de parâmetros flexível**: Otimize não só os valores, mas também quais indicadores usar.
- **Backtest, otimização, walk-forward e Monte Carlo** integrados.
- **Scripts automatizados para busca de estratégias vencedoras.**
- **Resultados detalhados e exportáveis.**

---

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone <url-do-repo>
   cd trading-backtest-system
   ```

2. **Crie e ative um ambiente virtual:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure suas API keys (se for usar dados da Binance):**
   - Defina as variáveis de ambiente `BINANCE_API_KEY` e `BINANCE_API_SECRET` ou edite seu `.env`.

---

## Estrutura dos Arquivos

- `config/settings.py`: Configuração centralizada da estratégia (`StrategyConfig`).
- `config/parameters.py`: Espaço de parâmetros para otimização.
- `src/`: Código-fonte principal (backtest, indicadores, otimização, etc).
- `run_backtest.py`: Script principal para rodar backtest, otimização, walk-forward e Monte Carlo.
- `auto_strategy_search.py`: Script para busca automática de estratégias vencedoras.
- `src/data/results/`: Resultados detalhados dos testes.

---

## Exemplo de Configuração (`StrategyConfig`)

```python
@dataclass
class StrategyConfig:
    # Trading
    operation_code: str = 'ETHBRL'
    asset_code: str = 'ETH'
    candle_interval: str = Client.KLINE_INTERVAL_5MINUTE
    quantity: float = 0.01
    take_profit_pct: float = 0.01
    stop_loss_pct: float = 0.007
    min_profit_to_sell: float = 0.002
    max_daily_loss_percentage: float = 0.10
    max_concurrent_positions: int = 1
    # Indicadores
    use_ema: bool = True
    ema_fast: int = 7
    ema_slow: int = 15
    use_rsi: bool = True
    rsi_period: int = 7
    rsi_oversold: int = 20
    rsi_overbought: int = 75
    use_macd: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_bollinger: bool = False
    bollinger_period: int = 20
    use_volume: bool = True
    volume_ma: int = 20
    volume_threshold: float = 1.3
    use_trend: bool = True
    min_trend_strength: float = 0.2
    # Sinais
    min_score: int = 65
```

---

## Como rodar o sistema

### 1. Backtest simples

```bash
python run_backtest.py
# Escolha a opção 1 no menu
```

### 2. Otimização de parâmetros

```bash
python run_backtest.py
# Escolha a opção 2 no menu
# Escolha o perfil de estratégia (Scalping, Swing, Default)
```

- O sistema irá otimizar tanto os valores quanto a ativação dos indicadores.

### 3. Walk-forward

```bash
python run_backtest.py
# Escolha a opção 3 no menu
```

### 4. Monte Carlo

```bash
python run_backtest.py
# Escolha a opção 4 no menu
```

### 5. Análise completa

```bash
python run_backtest.py
# Escolha a opção 5 no menu
```

### 6. Busca automática de estratégia vencedora

```bash
python auto_strategy_search.py
```

- O script irá rodar várias otimizações, ajustando ranges e ativação dos indicadores até encontrar uma estratégia vencedora.

---

## Como ativar/desativar indicadores

- Ajuste os campos `use_ema`, `use_rsi`, `use_macd`, `use_bollinger`, `use_volume`, `use_momentum` na sua configuração.
- O otimizador pode testar combinações de ativação automaticamente.

---

## Como adicionar novos indicadores

1. Implemente o cálculo no `TechnicalIndicators`.
2. Adicione o parâmetro de ativação e os parâmetros do indicador em `StrategyConfig` e no espaço de parâmetros.
3. Adapte o `SignalGenerator` para usar o novo indicador se ativado.

---

## Interpretação dos resultados

- Resultados detalhados são salvos em `src/data/results/`.
- Incluem métricas, equity curve, trades detalhados e configurações usadas.
- Use os arquivos `.json` e `.csv` para análise posterior.

---

## Testes

- Os testes usam a nova estrutura de configuração.
- Veja `tests/conftest.py` para exemplos de configs de teste.
- Para rodar os testes:

  ```bash
  pytest
  ```

---

## Troubleshooting

- **Erro de importação de indicadores/config:** Certifique-se de que está usando sempre `STRATEGY_CONFIG` e que todos os parâmetros necessários estão presentes.
- **Dados não carregam:** Verifique suas chaves de API e conexão com a internet.
- **Resultados ruins:** Tente ajustar os ranges de parâmetros, ativar/desativar indicadores, ou mudar o timeframe.

---

## Dúvidas ou contribuições

Abra uma issue ou envie um pull request!
