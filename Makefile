run:
	python src/csiro_biomass/main.py --config configs/train.yaml

format:
	uv run ruff format src/

check:
	uv run ruff check src/