#!/usr/bin/env bash
# Parent only. Use a new tag for each matched pair. Existing results stay intact.
set -euo pipefail
task=${1:?Use ordinary, candidate, qualify, or paced}
tag=${2:-trial1}
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
exp=/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments
work=/mnt/nvme-500/tmp/alpamayo-speed-20261116
source_root=$work/agent-paced-dynamic-source-cpu02/alpamayo
invocation=$work/agent-paced-dynamic-source-cpu02/pipeline-invocation.json
python=/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python
pythonpath=/tmp/opencode/alpamayo-asius-20261114/tinygrad:$source_root:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo
reference=$work/agent-startup-spec-reference/report.json
baseline=$work/agent-startup-cache-$tag-ordinary/report.json
candidate=$work/agent-startup-cache-$tag-candidate/report.json
case "$task" in
  ordinary|candidate)
    variant=ordinary
    if [[ "$task" == candidate ]]; then variant=serial-first-capture; fi
    exec python3 /tmp/opencode/run-alpamayo-speed-trial.py "agent-startup-cache-$tag-$task" \
      --program "$exp/agent-startup-cache-run.py" --timeout-seconds 1200 \
      --set variant "$variant" --set task replay --set base-source "$source_root" \
      --set invocation "$invocation" --set mode consumer --set decoder speculative \
      --set prefill-attention shuffle-fma --set compare-reference "$reference" \
      --flag execute-gpu --env PYTHONPATH "$pythonpath"
    ;;
  qualify)
    exec env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$pythonpath" DEV=CPU "$python" "$exp/agent-startup-cache-run.py" \
      --variant serial-first-capture --task qualify --base-source "$source_root" \
      --baseline-report "$baseline" --reference-report "$reference" --candidate-report "$candidate"
    ;;
  paced)
    exec python3 /tmp/opencode/run-alpamayo-speed-trial.py "agent-startup-cache-$tag-paced" \
      --program "$exp/agent-startup-cache-run.py" --timeout-seconds 1200 \
      --set variant serial-first-capture --set task paced --set base-source "$source_root" \
      --set invocation "$invocation" --set mode run --set baseline-report "$baseline" \
      --set reference-report "$reference" --set consumer-parity-report "$candidate" \
      --set hz 10 --set start-seconds 35 --set seconds 8 --set inference-budget-seconds 180 \
      --set raw-frames "$work/agent-paced-replay-cache-35-43/rgb" --set warp native \
      --env PYTHONPATH "$pythonpath"
    ;;
  *) echo 'Use ordinary, candidate, qualify, or paced' >&2; exit 2 ;;
esac
