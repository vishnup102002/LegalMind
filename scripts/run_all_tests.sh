#!/usr/bin/env bash
# LegalMind Automated CI/CD & QA Testing Pipeline
set -e

echo "--------------------------------------------------------"
echo "  Running LegalMind 3-Layer Automated Test Suite"
echo "--------------------------------------------------------"

./venv/bin/python3 tests/run_full_suite.py
