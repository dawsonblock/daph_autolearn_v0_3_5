.PHONY: install test dataset

install:
	pip install -e ".[dev]"

test:
	pytest -q

dataset:
	python scripts/generate_dataset.py --out data/math_v1.jsonl --n 4000
