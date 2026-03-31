#!/bin/bash
# Run backend pytest suite
cd "$(dirname "$0")/backend"
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate
echo "=== BasisFactory unit tests ==="
pytest tests/test_basis_factory.py -v
echo ""
echo "=== SCF Engine integration tests (requires PySCF) ==="
pytest tests/test_scf_engine.py -v --tb=short
