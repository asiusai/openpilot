#!/usr/bin/env bash
# Parent only. The existing trial wrapper owns GPU serialization and ACL cleanup.
set -euo pipefail
task=${1:?Use reference, candidate, qualify, or paced}
exp=/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments
work=/mnt/nvme-500/tmp/alpamayo-speed-20261116
source_root=$work/agent-paced-dynamic-source-cpu02/alpamayo
invocation=$work/agent-paced-dynamic-source-cpu02/pipeline-invocation.json
python=/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python
pythonpath=/tmp/opencode/alpamayo-asius-20261114/tinygrad:$source_root:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo
reference=$work/agent-startup-spec-reference/report.json
candidate=$work/agent-startup-spec-candidate/report.json
case "$task" in
  reference)
    exec python3 /tmp/opencode/run-alpamayo-speed-trial.py agent-startup-spec-reference \
      --program "$exp/agent-startup-spec-run.py" --timeout-seconds 1200 \
      --set task replay --set base-source "$source_root" --set invocation "$invocation" \
      --set mode reference --set decoder speculative --set prefill-attention shuffle-fma \
      --flag execute-gpu --env PYTHONPATH "$pythonpath"
    ;;
  candidate)
    exec python3 /tmp/opencode/run-alpamayo-speed-trial.py agent-startup-spec-candidate \
      --program "$exp/agent-startup-spec-run.py" --timeout-seconds 1200 \
      --set task replay --set base-source "$source_root" --set invocation "$invocation" \
      --set mode consumer --set decoder speculative --set prefill-attention shuffle-fma \
      --set compare-reference "$reference" --flag execute-gpu --env PYTHONPATH "$pythonpath"
    ;;
  qualify)
    exec env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$pythonpath" DEV=CPU "$python" "$exp/agent-startup-spec-run.py" \
      --task qualify --base-source "$source_root" --reference-report "$reference" --candidate-report "$candidate"
    ;;
  paced)
    exec python3 /tmp/opencode/run-alpamayo-speed-trial.py agent-startup-spec-paced \
      --program "$exp/agent-startup-spec-run.py" --timeout-seconds 1200 \
      --set task paced --set base-source "$source_root" --set invocation "$invocation" \
      --set mode run --set reference-report "$reference" --set consumer-parity-report "$candidate" \
      --set hz 10 --set start-seconds 35 --set seconds 8 --set inference-budget-seconds 180 \
      --set raw-frames "$work/agent-paced-replay-cache-35-43/rgb" --set warp native \
      --env PYTHONPATH "$pythonpath"
    ;;
  *) echo 'Use reference, candidate, qualify, or paced' >&2; exit 2 ;;
esac
