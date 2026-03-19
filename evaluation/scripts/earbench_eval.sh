#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TARGET_MODEL=${TARGET_MODEL:?Set TARGET_MODEL to a local checkpoint path or remote model name}
TARGET_API_URL=${TARGET_API_URL:-}
TARGET_API_KEY=${TARGET_API_KEY:-}
EVALUATION_API_URL=${EVALUATION_API_URL:?Set EVALUATION_API_URL to the judge endpoint}
EVALUATION_API_KEY=${EVALUATION_API_KEY:-EMPTY}
VERSION=${VERSION:-v1}
ADAPTER=${ADAPTER:-}

cd "${REPO_ROOT}"
export TARGET_API_URL TARGET_API_KEY EVALUATION_API_URL EVALUATION_API_KEY
python -m evaluation.eval_earbench --target_model "${TARGET_MODEL}" --version "${VERSION}" ${ADAPTER:+--adapter "${ADAPTER}"}
