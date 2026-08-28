# Four-PE Kernels

This directory contains kernels whose implementation explicitly depends on
four PEs. They use one of two execution models:

- Cooperative execution: shared matrix operands and four-PE `TMATMUL` for
  Matmul and FlashAttention.
- SPMD partitioning: `get_thread_idx()` assigns disjoint row ranges for TADD,
  TROWSUM and RMSNorm.

Current implementations:

| Operator | Header |
|---|---|
| TADD | `element_wise/tadd_multithread.hpp` |
| FlashAttention | `fa/fa_2d_unroll_gmma.hpp` |
| Matmul | `matmul/matmul_multithread.hpp` |
| Shared Matmul | `matmul/matmul_shared.hpp` |
| Shared-B-reuse Matmul | `matmul/matmul_shared_reuseB.hpp` |
| Low-precision Matmul | `matmul/matmul_shared_lowp.hpp` |
| RMSNorm | `normalization/rms_norm/rms_norm.hpp` |
| TROWSUM | `reduction/trowsum_multithread.hpp` |

## Run with gfrun

Run an ELF built from a multi-thread operator test with four simulated PEs:

```bash
gfrun -t 1 -s softcore.multiThreadNum=4 -f /path/to/operator.elf
```

Do not omit `softcore.multiThreadNum=4`: cooperative kernels require all four
PEs, while SPMD kernels use thread IDs 0 through 3 to partition the work.

The run passes when `gfrun` exits with status 0 and its output contains both
`Reach the End of Benchmark` and `R2 = 0`.
