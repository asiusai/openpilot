#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="20.6"
fi

export STAGING_ROOT="/data/safe_staging"

# Alpamayo planner on the external GPU (openpilot/selfdrive/alpamayo/README.md).
# ALPAMAYO_SHADOW=1 only records its plans. ALPAMAYO_PLANNER=0 turns it off.
export ALPAMAYO_CONFIG="${ALPAMAYO_CONFIG:-/data/alpamayo/config.json}"
export ALPAMAYO_KEEP_WARM="${ALPAMAYO_KEEP_WARM:-1}"
