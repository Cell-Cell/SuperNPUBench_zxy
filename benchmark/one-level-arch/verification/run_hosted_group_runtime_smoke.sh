#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ARCH_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
TEST_ROOT="$ARCH_ROOT/test"

: "${COMPILER_DIR:?set COMPILER_DIR to the Linx LLVM bin directory}"
: "${MUSL_SYSROOT:?set MUSL_SYSROOT to the installed Linx musl sysroot}"
: "${GFRUN:?set GFRUN to the gfrun executable}"

CLANG=${CLANG:-$COMPILER_DIR/clang}
if [ ! -x "$CLANG" ] && [ -x "$COMPILER_DIR/clang-23" ]; then
    CLANG="$COMPILER_DIR/clang-23"
fi
TARGET=${LINX_TARGET:-linx64-unknown-linux-musl}
OUT_DIR=${OUT_DIR:-$SCRIPT_DIR/output/hosted-group-runtime-smoke}

mkdir -p "$OUT_DIR"

"$CLANG" -target "$TARGET" --sysroot "$MUSL_SYSROOT" -D__linx \
    -fno-pie -O2 -Wall -Wextra -Werror \
    -I "$TEST_ROOT/common" \
    -c "$TEST_ROOT/common/src/linx_group_runtime.cpp" \
    -o "$OUT_DIR/linx_group_runtime.o"

"$CLANG" -target "$TARGET" --sysroot "$MUSL_SYSROOT" -D__linx \
    -fno-pie -O2 -Wall -Wextra -Werror \
    -I "$TEST_ROOT/common" \
    -c "$SCRIPT_DIR/hosted_group_runtime_smoke.cpp" \
    -o "$OUT_DIR/hosted_group_runtime_smoke.o"

"$CLANG" -target "$TARGET" \
    -c "$SCRIPT_DIR/hosted_group_runtime_start.s" \
    -o "$OUT_DIR/hosted_group_runtime_start.o"

"$CLANG" -target "$TARGET" -static -fuse-ld=lld -nostdlib \
    "$OUT_DIR/hosted_group_runtime_start.o" \
    "$OUT_DIR/hosted_group_runtime_smoke.o" \
    "$OUT_DIR/linx_group_runtime.o" \
    -Wl,--image-base=0x40000000 \
    -o "$OUT_DIR/hosted_group_runtime_smoke"

"$GFRUN" -f "$OUT_DIR/hosted_group_runtime_smoke" \
    -s softcore.multiThreadNum=4
