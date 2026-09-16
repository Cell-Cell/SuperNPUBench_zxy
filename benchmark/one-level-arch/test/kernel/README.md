# Test Kernel — Operator Test Suites

Per-operator test code and build scripts. Each operator directory has a
`Makefile`, a `compile.all` (typical configs), and `src/`. All suites include
the shared `common/Makefile.common`.

## Directory Structure

Single-thread suites have been retired. The active tree is the four-PE
(four-thread) suite:

```
test/kernel/
└── multi_thread/   broadcast  concat  conv2d  element_wise  fa  gather
                    matmul  mxquant  reduction  transpose  vec
```

See [`multi_thread/README.md`](multi_thread/README.md) for the operator list,
partition rules, and the gfrun / RES_CHECK regression status.

## Build

```bash
# single config
cd test/kernel/multi_thread/matmul
make TESTCASE=matmul COMPILER_DIR="$COMPILER_DIR" B=1 M=256 N=256 K=256 tM=32 tN=32 tK=32

# per-operator batch
cd test/kernel/multi_thread/matmul && bash compile.all

# whole one-level-arch backend (from repo root)
./compile_all.sh one-level
```

Build products are written under the arch-level `output/` directory
(`benchmark/one-level-arch/output/`), which is gitignored.

## Adding a Test

1. Create `test/kernel/multi_thread/<operator>/` with `src/`, `Makefile`,
   `compile.all`.
2. Add the operator to `compile_all.sh`.
3. `include` the shared `common/Makefile.common` (adjust the relative depth
   for the extra `multi_thread/` level).

## See Also
- [Top-level README](../../README.md)
- [Operator implementations](../kernels/README.md)
