# Single-PE Kernels

This directory contains the original one-level-architecture operator
implementations that execute on one PE. Operator subdirectories retain their
previous names, so includes now use the following form:

```cpp
#include "single_thread/matmul/matmul.hpp"
```

Four-PE-specific implementations are kept separately under
[`../multi_thread/`](../multi_thread/README.md).
