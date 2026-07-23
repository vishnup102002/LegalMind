#!/usr/bin/env bash
set -e

echo "=================================================================="
echo "      LegalMind Multi-Domain & Multi-Lingual Test Suite           "
echo "=================================================================="

# Run unit and Pydantic checks
./venv/bin/pytest tests/test_layer1_pydantic_units.py -v

# Run domain & language matrix tests
./venv/bin/pytest tests/test_layer3_diverse_legal_domains.py -v

echo "=================================================================="
echo "  ✅ ALL DOMAINS (LABOR, TENANCY, CONSUMER, CONTRACT) PASSED!     "
echo "=================================================================="
