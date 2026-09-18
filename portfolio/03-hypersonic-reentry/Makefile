# Hypersonic atmospheric reentry simulator -- one-command reproduction.
#
#   make all    trajectories -> validation -> figures -> hero.png + hero.mp4
#   make quick  same pipeline with a short video (pipeline smoke test)
#   make test   pytest
#
# No network access is required.  Install the pinned stack with
#   python3 -m pip install -r requirements.txt

PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: all quick test trajectories validate figures hero still clean help

all: ## full reproduction (~15 min on 4 cores)
	$(PYTHON) run_all.py

quick: ## full pipeline, short video -- for checking everything runs
	$(PYTHON) run_all.py --quick

test: ## run the automated test suite
	$(PYTHON) -m pytest

trajectories: ## nominal entries + entry-corridor sweeps -> results/
	$(PYTHON) scripts/run_trajectories.py

validate: ## the four numbered validations -> validation/
	cd validation && $(PYTHON) run_all_validation.py

figures: ## the six technical figures -> figures/
	$(PYTHON) scripts/make_figures.py

hero: ## media/hero.png + media/hero.mp4
	$(PYTHON) scripts/make_hero.py

still: ## media/hero.png only
	$(PYTHON) scripts/make_hero.py --still-only

clean: ## remove generated artefacts (keeps the source)
	rm -f results/*.json results/*.npz
	rm -f validation/*.json validation/*.txt
	rm -f figures/*.png
	rm -f media/hero.png media/hero.mp4 media/hero_manifest.json
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
