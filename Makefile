PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: venv install quick test

venv:
	$(UV) venv --clear .venv
	$(UV) pip install --python $(PYTHON) -r requirements.txt

install: venv

quick:
	$(PYTHON) eaqga.py --quick

test:
	$(PYTHON) tests/check_integrity.py
	$(PYTHON) tests/test.py
