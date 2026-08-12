# 证据：R2=6 返回值解码分析

## 编码机制

当 `podInfoMatch != podInfoTotal` 时，失败路径编码诊断信息：

```
ret = actualVal × 10000 + expectedPod0 × 1000 + tokenId0
       ─────────────────   ───────────────────   ──────────
       算子实际写入值        验证期望值             section 0 的
       （0 或 1）           （0 或 1）             首个 token id
                                                （0..511）
```

## 约束分析

`genTopkIndex` 生成 `expertId ∈ [0, 128)`。因此：

```
curDstPod = expertId / 64 ∈ {0, 1}
```

两个值均 `< kSuperPodNum (2)`，因此 `if (curDstPod < kSuperPodNum)` 在标准语义下**恒为真**。这意味着 `expectedPod0` **应恒为 1**。

若 `expectedPod0 = 1`，则：

```
ret = actualVal × 10000 + 1000 + tokenId0 ≥ 1000
```

**在标准语义下 R2=6 不可能成立**（最小值为 1000）。

## R2=6 的解码

R2=6 要求以下三个条件同时成立：

| 变量 | 值 | 含义 |
|---|---|---|
| `actualVal` | 0 | 算子写入 `tokenSuperPodInfo[0] = 0`（应为 1） |
| `expectedPod0` | 0 | 验证计算 `expectedPod[0] = 0`（应为 1） |
| `tokenId0` | 6 | section 0 的首个 token 是 token id 6 |

```
ret = 0 × 10000 + 0 × 1000 + 6 = 6  ✓
```

## 解释

`actualVal=0` 和 `expectedPod0=0` 都是错误的（应为 1），但二者错误的原因**不同**：

- **算子侧**（`>> 6`）：gfrun 的标量右移产生了错误结果 → `dstPodLocal` 未被正确设置 → `tokenSuperPodInfo` 写入 0
- **验证侧**（`/ 64`）：gfrun 的标量整数除法产生了错误结果 → `curDstPod ≥ kSuperPodNum` → `expectedPod` 保持 0

两侧在同一 `(s0=0, i0=0)` 位置恰好都产生 0 的巧合导致 `ret=6`。在其他位置两侧产生不同的错误值，导致 `podInfoMatch != podInfoTotal`。

## Token 6 的 expert 分布（确定性 LCG 数据）

```
expert[ 0] =  57  pod=0    expert[ 8] =  14  pod=0
expert[ 1] =  70  pod=1    expert[ 9] =  12  pod=0
expert[ 2] = 126  pod=1    expert[10] = 116  pod=1
expert[ 3] =  39  pod=0    expert[11] =  99  pod=1
expert[ 4] =  57  pod=0    expert[12] =  93  pod=1
expert[ 5] =  99  pod=1    expert[13] = 108  pod=1
expert[ 6] =  47  pod=0    expert[14] = 112  pod=1
expert[ 7] =  95  pod=1    expert[15] = 113  pod=1
```

pod 0 有 6 个 expert → `dstPodLocal[0] = 1`
pod 1 有 10 个 expert → `dstPodLocal[1] = 1`

标准语义：`tokenSuperPodInfo[0] = 1`，`tokenSuperPodInfo[1] = 1`。
gfrun：`tokenSuperPodInfo[0] = 0`（actualVal=0）。
