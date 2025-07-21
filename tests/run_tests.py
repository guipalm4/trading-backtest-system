#!/usr/bin/env python3
"""
Script para executar testes do Trading Backtest Pro
"""

import sys
import subprocess
import argparse
import os


def run_command(command, description):
    """Executa comando e mostra resultado"""
    print(f"\n🔄 {description}...")
    print(f"Executando: {command}")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {description} - SUCESSO")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ {description} - FALHA")
        if result.stderr:
            print(result.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Executar testes do Trading Backtest Pro')
    parser.add_argument('--type', choices=['all', 'unit', 'integration', 'performance'],
                        default='all', help='Tipo de teste a executar')
    parser.add_argument('--coverage', action='store_true', help='Gerar relatório de cobertura')
    parser.add_argument('--parallel', action='store_true', help='Executar testes em paralelo')
    parser.add_argument('--fast', action='store_true', help='Executar apenas testes rápidos')
    parser.add_argument('--lint', action='store_true', help='Executar linting')
    parser.add_argument('--format', action='store_true', help='Formatar código')

    args = parser.parse_args()

    print("🚀 Trading Backtest Pro - Executor de Testes")
    print("=" * 50)

    success = True

    # Verificar se pytest está instalado
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} encontrado")
    except ImportError:
        print("❌ pytest não encontrado. Instale com: pip install pytest")
        return 1

    # Formatação de código
    if args.format:
        success &= run_command("black src/ tests/", "Formatação com Black")
        success &= run_command("isort src/ tests/", "Organização de imports")

    # Linting
    if args.lint:
        success &= run_command("flake8 src/ tests/", "Linting com flake8")

    # Construir comando de teste
    test_command = "pytest tests/"

    if args.type == 'unit':
        test_command += ' -m "unit"'
    elif args.type == 'integration':
        test_command += ' -m "integration"'
    elif args.type == 'performance':
        test_command += ' -m "performance"'

    if args.fast:
        test_command += ' -m "not slow"'

    if args.parallel:
        test_command += ' -n auto'

    if args.coverage:
        test_command += ' --cov=src --cov-report=html --cov-report=term'

    test_command += ' -v'

    # Executar testes
    success &= run_command(test_command, f"Testes {args.type}")

    # Resultado final
    print("\n" + "=" * 50)
    if success:
        print("🎉 TODOS OS TESTES PASSARAM!")
        return 0
    else:
        print("💥 ALGUNS TESTES FALHARAM!")
        return 1


if __name__ == "__main__":
    sys.exit(main())