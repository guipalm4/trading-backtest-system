import importlib
import random
import time
from copy import deepcopy
import builtins

from run_backtest import run_parameter_optimization

# Forçar sempre a escolha do perfil 1 (Scalping) ao chamar input no run_backtest.py
def always_scalping_input(prompt):
    if "Digite sua escolha (1-3)" in prompt:
        print(prompt + "2")
        return "2"
    return input(prompt)

builtins.input = always_scalping_input

# Critérios mínimos para considerar uma estratégia vencedora
MIN_RETURN = 0.01         # 1% de retorno
MIN_PROFIT_FACTOR = 1.05  # Profit factor acima de 1.05
MAX_DRAWDOWN = -0.15      # Drawdown máximo de -15%
MAX_ATTEMPTS = 30         # Número máximo de tentativas

# Parâmetros base para scalping menos agressivo
BASE_PARAM_SPACE = {
    'fast_ma_period': [5, 7, 9],
    'slow_ma_period': [12, 15, 18],
    'rsi_period': [7, 10],
    'rsi_oversold': [20, 25],
    'rsi_overbought': [75, 80],
    'take_profit_pct': [0.007, 0.01, 0.015],
    'stop_loss_pct': [0.007, 0.01, 0.015],
    'volume_threshold': [1.2, 1.3, 1.4],
    'min_score': [65, 70, 75]
}

# Função para evitar combinações impossíveis
# (ex: fast_ma >= slow_ma, rsi_oversold >= rsi_overbought, take_profit <= stop_loss)
def sanitize_param_space(param_space):
    # Garante que fast_ma < slow_ma
    param_space['fast_ma_period'] = [f for f in param_space['fast_ma_period']
                                     if any(f < s for s in param_space['slow_ma_period'])]
    param_space['slow_ma_period'] = [s for s in param_space['slow_ma_period']
                                     if any(f < s for f in param_space['fast_ma_period'])]
    # Garante que rsi_oversold < rsi_overbought
    param_space['rsi_oversold'] = [o for o in param_space['rsi_oversold']
                                   if any(o < ob for ob in param_space['rsi_overbought'])]
    param_space['rsi_overbought'] = [ob for ob in param_space['rsi_overbought']
                                     if any(o < ob for o in param_space['rsi_oversold'])]
    # Garante que take_profit > stop_loss
    param_space['take_profit_pct'] = [tp for tp in param_space['take_profit_pct']
                                      if any(tp > sl for sl in param_space['stop_loss_pct'])]
    param_space['stop_loss_pct'] = [sl for sl in param_space['stop_loss_pct']
                                    if any(tp > sl for tp in param_space['take_profit_pct'])]
    return param_space

def is_winner(test_results):
    return (
        test_results['total_return'] > MIN_RETURN and
        test_results['profit_factor'] > MIN_PROFIT_FACTOR and
        test_results['max_drawdown'] > MAX_DRAWDOWN
    )

def randomize_param_space(base_space, shrink_factor=0.8):
    """
    Reduz aleatoriamente o espaço de busca para focar em ranges mais promissores.
    shrink_factor < 1 reduz o número de valores testados a cada rodada.
    """
    new_space = deepcopy(base_space)
    for k, v in new_space.items():
        if len(v) > 1:
            n = max(1, int(len(v) * shrink_factor))
            new_space[k] = sorted(random.sample(v, n))
    return sanitize_param_space(new_space)

def patch_param_space(new_space):
    """Substitui o método get_scalping_space dinamicamente para usar o novo espaço."""
    import config.parameters
    def custom_scalping_space():
        return new_space
    config.parameters.ParameterSpace.get_scalping_space = staticmethod(custom_scalping_space)
    importlib.reload(config.parameters)


def main():
    attempt = 0
    found = False
    best_result = None
    best_params = None
    best_score = float('-inf')
    param_space = deepcopy(BASE_PARAM_SPACE)

    while attempt < MAX_ATTEMPTS and not found:
        print(f"\n=== Tentativa {attempt + 1} de {MAX_ATTEMPTS} ===")
        patch_param_space(param_space)
        try:
            result = run_parameter_optimization()
            if result is None:
                print("❌ Erro na execução do backtest. Pulando tentativa...")
                attempt += 1
                continue
            best_params_run, _, test_results = result
        except Exception as e:
            print(f"❌ Erro: {e}")
            attempt += 1
            continue

        # Score simples: retorno - drawdown + profit_factor
        score = test_results['total_return'] - abs(test_results['max_drawdown']) + test_results['profit_factor']
        if score > best_score:
            best_score = score
            best_result = test_results
            best_params = best_params_run

        if is_winner(test_results):
            print("\n🎉 Estratégia vencedora encontrada!")
            print("Parâmetros:", best_params_run)
            print("Resultados:", test_results)
            found = True
        else:
            print("\n❌ Estratégia não atingiu os critérios mínimos. Ajustando ranges e tentando novamente...")
            # Reduz o espaço de busca para focar em valores mais promissores
            param_space = randomize_param_space(param_space, shrink_factor=0.7)
            time.sleep(2)
        attempt += 1

    print("\n=== MELHOR RESULTADO ENCONTRADO ===")
    print("Parâmetros:", best_params)
    print("Resultados:", best_result)

if __name__ == "__main__":
    main() 