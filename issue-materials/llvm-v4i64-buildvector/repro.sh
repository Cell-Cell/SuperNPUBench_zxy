#!/bin/bash
# One-shot matrix: expects CXX (linx clang++) and INC (tileop-api include flags)
: "${CXX:?export CXX=<linx_blockisa_llvm_musl>/bin/clang++}"
: "${INC:?export INC=-I<tileop-api-include-dir>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
COMMON="-std=c++20 -D__linx -DENABLE_TENSOR_INSTR $INC"

run() { # $1=src $2=opts $3=label
    $CXX -c $2 $COMMON "$DIR/src/$1" -o /tmp/v4i64_repro.o > /tmp/v4i64_build.log 2>&1
    local ec=$?
    local sig=$(grep -m1 -o 'Cannot select: [^ ]*' /tmp/v4i64_build.log)
    [ -z "$sig" ] && sig=$(grep -m1 -o 'BGPR multi set!' /tmp/v4i64_build.log)
    [ -z "$sig" ] && sig="OK"
    printf '%-28s %-22s exit=%-3s %s\n' "$1" "$2" "$ec" "$sig"
}

run min_repro.cpp           "-mlxbc -O2"
run min_repro_4stores.cpp   "-mlxbc -O2"
run min_repro_volatile.cpp  "-mlxbc -O2"
run min_repro.cpp           "-mlxbc -O0"
