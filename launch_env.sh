#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -f /ASIUS ]; then
  export NO_IMU=1
  export NO_FAN_CONTROL=1
  export DEVICE_TYPE=v1
  export LD_LIBRARY_PATH="/opt/qcom-adreno/lib:/opt/qcom-adreno/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="20.6"
fi

if [ -f /ASIUS ] && [ -z "$VAMOS_VERSION" ]; then
  export VAMOS_VERSION="18.1"
fi

export STAGING_ROOT="/data/safe_staging"
