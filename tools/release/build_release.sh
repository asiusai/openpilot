#!/usr/bin/env bash
set -e
set -x

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "$DIR"

SOURCE_DIR="$(git rev-parse --show-toplevel)"
ARTIFACTS_DIR="${PREBUILT_ARTIFACTS:-}"
if [ -n "$ARTIFACTS_DIR" ]; then
  test -f /ASIUS
  test "$(uname -m)" = aarch64
  mkdir -p "$ARTIFACTS_DIR"
  ARTIFACTS_DIR="$(realpath "$ARTIFACTS_DIR")"
  BUILD_DIR="$(mktemp -d "${RUNNER_TEMP:-/tmp}/openpilot-prebuilt.XXXXXX")"
  trap 'rm -rf "$BUILD_DIR"' EXIT
else
  BUILD_DIR=/data/openpilot
fi

export PYTHONPATH="$BUILD_DIR:$BUILD_DIR/msgq_repo:$BUILD_DIR/opendbc_repo:$BUILD_DIR/rednose_repo:$BUILD_DIR/teleoprtc_repo:$BUILD_DIR/tinygrad_repo"

if [ -z "$RELEASE_BRANCH" ] && [ -z "$ARTIFACTS_DIR" ]; then
  echo "RELEASE_BRANCH is not set"
  exit 1
fi

BUILD_BRANCH=release-mici-staging


# set git identity
source "$DIR/identity.sh"

echo "[-] Setting up repo T=$SECONDS"
if [ -n "$ARTIFACTS_DIR" ]; then
  # The artifact has its own Git objects and no source history or release branch.
  git init "$BUILD_DIR"
  cd "$BUILD_DIR"
  EMPTY_TREE="$(git hash-object -t tree -w --stdin </dev/null)"
  git commit-tree "$EMPTY_TREE" -m "Prebuilt artifact" > .git/HEAD
else
  if ! git -C "$SOURCE_DIR" worktree remove --force "$BUILD_DIR" 2>/dev/null; then
    rm -rf "$BUILD_DIR"
  fi
  git -C "$SOURCE_DIR" worktree prune
  git -C "$SOURCE_DIR" worktree add --detach --no-checkout "$BUILD_DIR"
  cd "$BUILD_DIR"
  git update-ref -d "refs/heads/$BUILD_BRANCH"
  git symbolic-ref HEAD "refs/heads/$BUILD_BRANCH"
  git read-tree --empty
fi

# do the files copy
echo "[-] copying files T=$SECONDS"
cd "$SOURCE_DIR"
./tools/release/release_files.py | xargs -0 cp -pR --parents -t "$BUILD_DIR" --

# in the directory
cd "$BUILD_DIR"
source "$SOURCE_DIR/tools/release/setup_lfs.sh"

# use the full CPU available for speeding up the build.
# openpilot resets the CPU frequencies when test_onroad.py runs below.
for policy in /sys/devices/system/cpu/cpufreq/policy*; do
  [ -d "$policy" ] || continue
  hardware_max="$(cat "$policy/cpuinfo_max_freq")"
  echo "$hardware_max" | sudo tee "$policy/scaling_max_freq" >/dev/null
done

scons ${SCONS_JOBS:+-j"$SCONS_JOBS"}

if [ -z "$PANDA_DEBUG_BUILD" ]; then
  # release panda fw
  CERT=/data/pandaextra/certs/release RELEASE=1 scons panda/
else
  # build with ALLOW_DEBUG=1 to enable features like experimental longitudinal
  scons panda/
fi

if [ -n "$INCLUDE_BIG_MODEL" ]; then
  python3 -c 'from openpilot.selfdrive.modeld.helpers import chestnut_compiled; assert chestnut_compiled()'
fi

# Ensure no submodules in release
if [ -n "$(git ls-files --stage | awk '$1 == "160000"')" ]; then
  echo "submodules found:"
  git submodule status
  exit 1
fi
git submodule status

# Cleanup
find . -name '*.a' -delete
find . -name '*.o' -delete
find . -name '*.os' -delete
find . -name '*.pyc' -delete
find . -name '__pycache__' -delete
rm -rf .sconsign.dblite tools/release/ .cache/
rm -f openpilot/selfdrive/modeld/models/*.onnx*

# Mark as prebuilt release
touch prebuilt

if [ -n "$ARTIFACTS_DIR" ]; then
  python3 "$SOURCE_DIR/tools/release/package_prebuilt.py" "$SOURCE_DIR" "$BUILD_DIR" "$ARTIFACTS_DIR"
  exit 0
fi

VERSION=$(cat openpilot/common/version.h | awk -F[\"-]  '{print $2}')
# Add built files to git
# writing larger objects is faster than compressing them on-device
git -c core.compression=0 add -f .
git -c core.compression=0 -c gc.auto=0 commit -m "openpilot v$VERSION"
source "$SOURCE_DIR/tools/release/check_file_sizes.sh"

# Run tests
cd "$BUILD_DIR"
RELEASE=1 ./openpilot/selfdrive/test/test_onroad.py "$@"
#tools/test_runner.py openpilot/selfdrive/car/tests/test_car_interfaces.py

echo "[-] pushing release T=$SECONDS"
REFS=()
for branch in ${RELEASE_BRANCH//,/ }; do
  REFS+=("$BUILD_BRANCH:$branch")
done
# uploading the larger pack is faster than spending CPU to optimize it
# The big model is already published to LFS by the source branch.
GIT_LFS_SKIP_PUSH=1 git -c pack.window=0 -c pack.depth=0 -c pack.compression=0 push -f origin "${REFS[@]}"

echo "[-] done T=$SECONDS"
