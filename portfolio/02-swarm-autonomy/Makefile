PY ?= python3

.PHONY: all quick test hero figures validate clean help

help:
	@echo "make all       full reproduction: scenarios -> sweep -> validation -> figures -> hero"
	@echo "make quick     short version (small scenarios, still image only)"
	@echo "make test      run the pytest suite"
	@echo "make validate  run the validation scripts only"
	@echo "make figures   rebuild figures/ from existing results/"
	@echo "make hero      rebuild media/hero.png and media/hero.mp4"
	@echo "make clean     remove generated artefacts"

all:
	$(PY) run_all.py

quick:
	$(PY) run_all.py --quick

test:
	$(PY) -m pytest

validate:
	cd validation && $(PY) validate_dynamics.py && $(PY) validate_tracking.py \
	  && $(PY) validate_avoidance.py && $(PY) validate_regression.py \
	  && $(PY) validate_turbulence.py

figures:
	$(PY) make_figures.py

hero:
	$(PY) make_hero.py

clean:
	rm -rf results/*.pkl results/*.json figures/*.png media/hero.png \
	       media/hero.mp4 media/hero_thumb.png validation/*.json \
	       validation/tracking_curves.npz .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
