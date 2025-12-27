test:
	pytest --tb=no --disable-warnings

run:
	python src/csiro_biomass/main.py --config configs/train.yaml

format:
	uv run ruff check --fix src/

check:
	uv run ruff check src/

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/csiro_biomass/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf models/*
	rm -rf wandb/*