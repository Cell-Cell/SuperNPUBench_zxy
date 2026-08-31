# LinxV5 后端 BUILD_VECTOR 合并 store 崩溃（v4i64 新变体 + v2i64 已知家族）最小复现

9 行用户代码：8 个连续 i32 零 store（经 `core*2+e` 计算索引）被后端合并为 32B
BLK_TSTORE，生成 `v4i64 = BUILD_VECTOR`，ISel 报 `Cannot select` 崩溃。

## 矩阵（同一 src/，仅换文件/选项）

| 变体 | 选项 | 结果 |
|---|---|---|
| `src/min_repro.cpp`（8×i32 零 store） | `-mlxbc -O2` | ❌ `error in backend: Cannot select: v4i64 = BUILD_VECTOR`（**新变体：4×i64 = 32B**） |
| `src/min_repro_4stores.cpp`（4×i32 零 store） | `-mlxbc -O2` | ❌ `Cannot select: v2i64 = BUILD_VECTOR`（已知家族的最小化形态） |
| `src/min_repro_volatile.cpp`（volatile 数组） | `-mlxbc -O2` | ✅ 编译通过（当前规避手段） |
| `src/min_repro.cpp` | `-mlxbc -O0` | ❌ 另一断言：`LinxV5TRegToOffset.cpp:906 __verifyGPR "BGPR multi set!"` |

不需要任何 tile API / -mllvm 开关：裸 `-mlxbc -O1/-O2` 即触发（证据见
`evidence/`；`-mllvm -linxv5-enable-continuous-mem-opt=false` 显式关闭也不影响，
说明合并发生在通用 store 合并 / LinxV5 lowering，而非该 pass 专属）。

## 一键复现

```bash
export CXX=<linx_blockisa_llvm_musl>/bin/clang++
export INC="-I<tileop-api include 所在> -I<any kernels include>"
bash repro.sh
```

## 真实触发上下文

PTO-ISA/SuperNPUBench PR #74 新增的 `mega_moe_sim_mt`（真 4PE 分片 MoE 算子）
统计槽清零循环：每 PE 清 `stats[core*2+e]`（4 伪核 × 2 expert = 8 个连续 i32
零 store），编译即崩；修复 = stats 指针改 volatile（与该仓长期使用的
tokRef/tilingData i64 volatile 规避同族）。见 `kernel-context/`。

## 版本

| 组件 | Commit |
|---|---|
| llvm-project (LinxV5) | `611105f2be11`（dev-llvm15_56，B.IOT/B.IOS ADR 0069） |
| Linx-TileOP-API | `a795b97` |
| linx-toolchain-build | `e6a31ef`（clang 15.0.4，linx64v5-unknown-linux-musl） |
