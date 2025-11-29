#!/bin/bash
# Wrapper script to run SwinUNETR tests with proper PYTHONPATH setup
# This ensures the paths are set before Python/pytest starts
#
# Usage:
#   ./run_swin_unetr_tests.sh cell_observatory_finetune/tests/models/test_swin_unetr.py -v

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/cell_observatory_platform:${PYTHONPATH}"

# Run pytest with all arguments passed through
# If no arguments provided, default to running the swin_unetr tests
if [ $# -eq 0 ]; then
    # exec pytest cell_observatory_finetune/tests/models/test_swin_unetr.py -v
    exec pytest cell_observatory_finetune/tests/models/test_swin_unetr_integration.py -v
else
    exec pytest "$@"
fi

