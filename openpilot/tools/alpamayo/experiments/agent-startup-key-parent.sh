#!/usr/bin/env bash
# Parent only. No old reports, sources, or compiler cache entries are replaced.
set -euo pipefail
task=${1:?Use audit or replay}
tag=${2:-fix1}
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
exp=/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments
work=/mnt/nvme-500/tmp/alpamayo-speed-20261116
source_root=$work/agent-paced-dynamic-source-cpu02/alpamayo
pythonpath=/tmp/opencode/alpamayo-asius-20261114/tinygrad:$source_root:/tmp/opencode/openpilot-alpamayo:/home/john/asius/openpilot/opendbc_repo
flags=()
case "$task" in
  audit) flags=(--flag audit-only);;
  replay) ;;
  *) echo 'Use audit or replay' >&2; exit 2;;
esac
exec python3 /tmp/opencode/run-alpamayo-speed-trial.py "agent-startup-key-$tag-$task" \
  --program "$exp/agent-startup-key-run.py" --timeout-seconds 600 \
  --set task replay --set base-source "$source_root" \
  --set invocation "$work/agent-paced-dynamic-source-cpu02/pipeline-invocation.json" \
  --set mode consumer --set decoder speculative --set prefill-attention shuffle-fma \
  --set compare-reference "$work/agent-startup-spec-reference/report.json" \
  --set ordinary-report "$work/agent-startup-cache-trial1-ordinary/report.json" \
  --set key-proof "$work/agent-startup-key-proof.json" \
  "${flags[@]}" --flag execute-gpu --env PYTHONPATH "$pythonpath"
