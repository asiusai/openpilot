#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
bundle=${ALPAMAYO_BUNDLE:?Set ALPAMAYO_BUNDLE to the prepared bundle directory}
target=${ALPAMAYO_DEPLOY_TARGET:?Set ALPAMAYO_DEPLOY_TARGET to the SSH destination}
expected_serial=${ALPAMAYO_EXPECTED_SOC_SERIAL:?Set ALPAMAYO_EXPECTED_SOC_SERIAL to the hexadecimal SoC serial}
[[ "$expected_serial" =~ ^(0[xX])?[0-9a-fA-F]+$ ]]
ssh -o BatchMode=yes -o ConnectTimeout=6 "$target" python3 - "$expected_serial" <<'PY'
from pathlib import Path
import shutil
import sys
s=Path('/sys/devices/soc0/serial_number').read_text().strip().lower()
n=int(s,16) if s.startswith('0x') or any(c in 'abcdef' for c in s) else int(s)
assert n==int(sys.argv[1],16), 'Wrong Dragon'
assert shutil.disk_usage('/data').free > 14*1024**3, 'Need at least 14 GiB free'
print('Verified target Dragon; sufficient storage')
PY
rsync -a --partial --info=progress2 "$bundle/" "$target:/data/alpamayo-eval/"
ssh "$target" 'cd /data/alpamayo-eval && sha256sum -c SHA256SUMS && uv sync --locked --no-dev --project upstream && uv pip install --python /data/openpilot/.venv/bin/python --target capture-deps --no-deps opencv-python-headless==4.13.0.92'
