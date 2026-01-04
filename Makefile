.PHONY: setup demo test clean help

help:
	@echo "Available commands:"
	@echo "  make setup  - Install dependencies with uv"
	@echo "  make demo   - Run demo inference"
	@echo "  make test   - Run tests"
	@echo "  make clean  - Clean cache and temporary files"

setup:
	uv sync

demo:
	@if [ -z "$(INPUT)" ]; then \
		echo "⚠️  Please provide an input image:"; \
		echo "   make demo INPUT=path/to/your/diagram.png"; \
		exit 1; \
	fi
	mkdir -p runs/demo
	uv run archlens analyze "$(INPUT)" --output runs/demo

test:
	uv run pytest -q

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf **/__pycache__
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

