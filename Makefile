PROJECT = boreal_LOA

VENV=.venv
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip3
WHEEL=dist/*.whl

.PHONY: all
all: dist

.PHONY: help
help:
	@echo "help: display this help"
	@echo "version: update the version (for maintainers only)"
	@echo "test|tests: run the tests"
	@echo "clean: clean build artifacts"
	@echo "clean-venv: delete the virtual environment"
	@echo "clean-all|cleanall: delete build artifacts and the virtual environment"

dist: $(VENV)
	$(PIP) install build && $(PYTHON) -m build

$(WHEEL): dist

.PHONY: tag version
tag version:
	@echo "New version (form x.y.z)? " && read version && sed -e "s/__version__/$$version/" setup.cfg.template > setup.cfg && echo "git add setup.cfg && git commit -m 'Version set to $$version' && git tag v$$version && git push && git push --tag"

$(VENV):
	python3 -m venv $(VENV) && $(PIP) install --upgrade pip && $(PIP) install --upgrade twine

.PHONY: test tests
test tests:
	$(PYTHON) tests.py

.PHONY: clean-venv
clean-venv:
	-rm -rf $(VENV)

.PHONY: clean
clean:
	-rm -f *~
	-rm -rf __pycache__ *.pyc
	-rm -rf build dist $(PROJECT).egg-info

.PHONY: clean-all cleanall
clean-all cleanall: clean clean-venv

