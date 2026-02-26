.PHONY: install-hooks lint typecheck test check

install-hooks:
	@git config core.hooksPath scripts/hooks
	@echo "Git hooks activated (core.hooksPath -> scripts/hooks)."

lint:
	uv run ruff check src/ tests/

test:
	uv run python -m pytest -m "not slow" -q

typecheck:
	uv run pyright src/agm/

check: lint typecheck test
