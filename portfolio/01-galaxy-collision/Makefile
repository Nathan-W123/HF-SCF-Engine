# Gravitational N-body galaxy collision -- full reproduction.
#
#   make all    simulate -> validate -> figures -> hero   (~20 min, 4 cores)
#   make quick  the same pipeline at reduced N            (~3 min)
#   make test   the pytest suite                          (~2 min)
#   make clean  remove generated output

PYTHON ?= python3

.PHONY: all quick test figures hero sim validate summary clean distclean help

all:
	$(PYTHON) run_all.py

quick:
	$(PYTHON) run_all.py --quick

test:
	$(PYTHON) -m pytest

sim:
	$(PYTHON) run_all.py --only simulate

validate:
	$(PYTHON) run_all.py --only kepler,tree,isolated,conservation,morphology,summary

figures:
	$(PYTHON) run_all.py --only figures

hero:
	$(PYTHON) run_all.py --only hero

summary:
	$(PYTHON) run_all.py --only summary

clean:
	rm -rf figures/*.png media/*.png media/*.mp4 results/*.json \
	       validation/*.json .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf data

help:
	@echo "make all | quick | test | sim | validate | figures | hero | clean"
