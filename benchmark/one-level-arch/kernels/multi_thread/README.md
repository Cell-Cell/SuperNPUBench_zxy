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
| RMSNorm | `normalization/rms_norm/rms_norm_pto.hpp` |
| TROWSUM | `reduction/trowsum_multithread.hpp` |

Includes use the execution-model prefix, for example:

```cpp
#include "multi_thread/matmul/matmul_shared.hpp"
```
