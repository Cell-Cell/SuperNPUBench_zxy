# Kernels — Operator Implementations

Header-only PTO operator implementations, organized by execution model:

- [`multi_thread/`](multi_thread/README.md): four-PE kernels. Each operator
  directory holds the four-PE wrapper plus the single-PE base implementation it
  partitions, the latter under `<op>/detail/`. Shared helpers live in
  `multi_thread/utils/`.
- [`solution/`](../test/solution): fused / end-to-end operator solutions built
  on top of the kernel primitives.

## multi_thread/ operators

| Operator | Directory |
|---|---|
| Broadcast | `multi_thread/broadcast/` |
| Concat (gather / scatter) | `multi_thread/concat/` |
| 1x1 Conv2D | `multi_thread/conv2d/` |
| GELU / TADD | `multi_thread/element_wise/` |
| FlashAttention | `multi_thread/fa/` |
| Gather | `multi_thread/gather/` |
| Shared / low-precision Matmul | `multi_thread/matmul/` |
| MX-quantized Matmul | `multi_thread/mxquant/` |
| Row Cumsum / Max / Prod / Sum | `multi_thread/reduction/` |
| 2D Transpose | `multi_thread/transpose/` |
| SPMD partition / layout helpers | `multi_thread/utils/` |

See [`multi_thread/README.md`](multi_thread/README.md) for the partition rules,
the base/wrapper split, and the gfrun + RES_CHECK regression status.

## solution/ operators

`gather_v2`, `group_token_old`, `group_token_vec`, `matmul_test`, `mega_moe`,
`moe_combine`, `moe_dispatch`, `normalization` (RMSNorm / GroupNorm grad),
`qli`, `quant`, `quant_batch_matmul`, `quant_sparse_flash_mla`, `view_copy`.

## Design Principles
1. **Header-only** — easy integration/reuse.
2. **PTO paradigm** — unified tile-operation interface.
3. **Templated** — type and dimension parameterization.
4. **Optimization-oriented** — multiple variants per scenario.

## Usage

```cpp
#include "multi_thread/matmul/matmul_shared.hpp"
matmul_shared<float, gM, gN, gK, tM, tN, tK>(c_ptr, a_ptr, b_ptr);
```

## See Also
- [Top-level README](../../README.md)
- [Test suites](../test/kernel/README.md)
