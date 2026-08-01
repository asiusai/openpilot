#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "${DEVICE_TYPE:-}" ] && [ -r /sys/firmware/devicetree/base/model ]; then
  hardware_model="$(tr -d '\000' < /sys/firmware/devicetree/base/model)"
  case "$hardware_model" in
    "comma "*|"asius "*) export DEVICE_TYPE="${hardware_model#* }" ;;
    "Radxa Dragon Q6A") export DEVICE_TYPE="v1" ;;
    *) export DEVICE_TYPE="$hardware_model" ;;
  esac
fi

if [ "${DEVICE_TYPE:-}" = "v1" ]; then
  export NO_FAN_CONTROL=1
  export LD_LIBRARY_PATH="/opt/qcom-adreno/lib:/opt/qcom-adreno/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="20.6"
fi

if [ "${DEVICE_TYPE:-}" = "v1" ] && [ -z "$VAMOS_VERSION" ]; then
  export VAMOS_VERSION="18.1"
fi

export STAGING_ROOT="/data/safe_staging"
