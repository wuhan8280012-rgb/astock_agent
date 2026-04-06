#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PATTERN='(sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|TUSHARE_TOKEN=[0-9a-f]{20,}|AWS_SECRET_ACCESS_KEY=[A-Za-z0-9+/]{20,})'

echo "Scanning for potential secrets..."
if rg -n --hidden -g '!*.tar.gz' -g '!release/**' -g '!.venv/**' -g '!backtest/cache/**' "${PATTERN}" .; then
  echo "Potential secrets detected."
  exit 1
fi

echo "No obvious secrets found."
