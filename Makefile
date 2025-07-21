.PHONY: test test-unit test-integration test-performance test-coverage clean lint format

# Executar todos os testes
test:
	pytest tests/ -v

# Testes unitários apenas
test-unit:
	pytest tests/ -m "unit" -v

# Testes de integração apenas
test-integration:
	pytest tests/ -m "integration" -v

# Testes de performance
test-performance:
	pytest tests/ -m "performance" -v

# Testes com cobertura detalhada
test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Testes rápidos (sem os lentos)
test-fast:
	pytest tests/ -m "not slow" -v

# Testes paralelos
test-parallel:
	pytest tests/ -n auto -v

# Linting
lint:
	flake8 src/ tests/
	mypy src/

# Formatação de código
format:
	black src/ tests/
	isort src/ tests/

# Limpeza
clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Instalar dependências de teste
install-test-deps:
	pip install -r tests/requirements-test.txt

# Setup completo para desenvolvimento
setup-dev: install-test-deps
	pre-commit install
	@echo "Ambiente de desenvolvimento configurado!"

# Executar todos os checks de qualidade
check-all: lint test-coverage
	@echo "Todos os checks passaram!"