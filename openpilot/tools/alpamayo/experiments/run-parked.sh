#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
test "$#" -eq 0
test "$(id -un)" = comma
test -x /data/openpilot/.venv/bin/python
export PYTHONPATH=/data/openpilot:/data/openpilot/opendbc_repo:/data/openpilot/msgq_repo
out="$PWD/results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
/data/openpilot/.venv/bin/python preflight.py | tee "$out/preflight.json"
service=/run/runit/service/openpilot
test -d "$service"
sudo -n true
enabled_cpus=()
restore() {
  for path in "${enabled_cpus[@]}"; do printf '0\n' | sudo -n tee "$path" >/dev/null || true; done
  sudo -n sv up "$service"
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sudo -n sv -w 30 down "$service"
if pgrep -f 'selfdrive/modeld/modeld|selfdrive/manager/manager.py' >/dev/null; then
  echo 'Installed model/manager still running; aborting' >&2
  exit 1
fi
cat /sys/devices/system/cpu/online > "$out/cpus-before.txt"
for path in /sys/devices/system/cpu/cpu[1-9]*/online; do
  if test "$(cat "$path")" = 0; then
    enabled_cpus+=("$path")
    printf '1\n' | sudo -n tee "$path" >/dev/null
  fi
done
taskset -pc "$(cat /sys/devices/system/cpu/online)" $$ > "$out/cpu-affinity.txt"
cat /sys/devices/system/cpu/online > "$out/cpus-during.txt"
export PYTHONPATH="$PWD/tinygrad:$PWD/alpamayo"
export DEV=USB+AMD:LLVM GMMU=0 AM_POWER_LIMIT=100 TC_OPT=2
export CACHEDB="$PWD/cache.db" LOG_ROOT="$out/log" COMMA_CACHE="$out/cache" TMPDIR="$out/tmp"
mkdir -p "$LOG_ROOT" "$COMMA_CACHE" "$TMPDIR"
python="$PWD/upstream/.venv/bin/python"
unset ALPAMAYO_USB_SERIAL_UPLOAD ALPAMAYO_USB_PCIE
timeout -k 5 90 "$python" -m unittest test.external.external_test_usb_asm24.TestUSBIntegrity.testBoundaries -f -v > "$out/usb-integrity.log" 2>&1
printf '%s\n' 'Harald native USB transport' > "$out/transport.txt"
benchmark=alpamayo/benchmark.py
if test -n "${ALPAMAYO_FP8_CACHE:-}"; then
  benchmark=benchmark-fast.py
  cp "$ALPAMAYO_FP8_CACHE/index.json" "$out/weights-index.txt"
  printf '%s\n' 'Prequantized bulk loader; unchanged Harald inference' >> "$out/transport.txt"
fi
git -C /data/openpilot rev-parse HEAD > "$out/installed-openpilot.txt"
uname -a > "$out/kernel.txt"
count=0
for sample in ${ALPAMAYO_SAMPLES:-5-original 5-narrow30 25-original 25-narrow30 45-original 45-narrow30}; do
  case "$sample" in 5-original|5-narrow30|25-original|25-narrow30|45-original|45-narrow30) ;; *) exit 2 ;; esac
  count=$((count + 1))
  timeout -k 10 "${ALPAMAYO_CASE_TIMEOUT:-600}" "$python" "$benchmark" --checkpoint "$PWD/model" --input "$PWD/samples/$sample.npz" \
    --precision fp8 --greedy --vision-batch 2 --diffusion-steps 10 --gpu-timeout-ms 120000 \
    --warmup "${ALPAMAYO_WARMUP:-2}" --runs "${ALPAMAYO_RUNS:-1}" --output "$out/$sample.json" > "$out/$sample.log" 2>&1
done
"$python" summarize.py "$out" --expected "$count" > "$out/summary.log"
echo "Results: $out"
