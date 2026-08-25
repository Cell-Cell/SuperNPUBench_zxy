# [gfsim] moe_dispatch 死锁：FRET.STK 重定向 resolveTarget=0 → va=0 取指 → BHC pa=0 永久 stall（无 miss、无超时、无报错）

## 一、现象（一句话）

`moe_dispatch_v2` 算子（PTO 移植版，附 ELF 与源码）在 **gfrun 功能模型上 R2=0 全过**（164,205 条指令 / 21,195 个 block 正常执行），但在 **gfsim 时序模型上必然死锁**（exit 134）。死锁点固定：`C:40186, B199 STID0 BPC 0x11468 [STD FALL] STD flushed`。

- **非算子问题**：同一 ELF gfrun 精度/功能全对（下方 §3 对照），程序无死循环；
- **决定性死锁**：非阈值误报，`waitCycles` 4000 周期内无任何新 block 提交（`verified minsts=0`）；
- **与 #344 / PR #356 无关**：已在 main@045c224 基座 cherry-pick PR #356 验证，死锁依旧（PR #356 修复的 A3 store-gate 标量死锁与本 issue 机制不同，见 §6）。

## 二、版本信息

| 组件 | 分支 / Commit |
|---|---|
| SuperScalarModel（死锁主体） | `main` @ **`045c224`**（Merge PR #321，干净基座无本地修改） |
| 算子 | `moe_dispatch_v2`（SuperNPUBench `ops-20260823` 配对工具链编译，源码见材料包 `kernels/`） |
| 复现输入 | 材料包 `moe_dispatch_v2.elf`（14,576 字节） |

## 三、复现步骤（3 条命令）

```bash
git clone https://github.com/LinxISA/SuperScalarModel.git && cd SuperScalarModel
git checkout 045c224 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make gfsim gfrun -j
./bin/gfsim -f /path/to/moe_dispatch_v2.elf        # 预期: Deadlock, exit 134
./bin/gfrun -t 1 -f /path/to/moe_dispatch_v2.elf   # 对照: R2 = 0 正常完成（证明非算子问题）
```

**gfsim 输出（关键行）：**

```
[C:36130][TMA.NA]: nuke_pair_diag ld_stid=0 ld_bid=195/w0 ld_rid=54 st_stid=0 st_bid=190/w0 st_rid=22
[C:36131][TMA.NA]: nuke_pair_diag ld_stid=0 ld_bid=195/w0 ld_rid=54 st_stid=0 st_bid=190/w0 st_rid=20
...
[C:40185][DFX.NA]: Deadlock detected at thread 0, cycles=40186, waitCycles exceeded threshold, verified blocks=12487, verified minsts=0
ERROR:Deadlock execution at:
 thread: 0B199 STID0 BPC 0x11468 [STD FALL] STD flushed
    Cycle:40186  Error BlockID:199  Wait cycles:3999
    Ref BPC: 0x11d98 isPar: 0
    Model BPC: 0x11468 isPar: 0 completed: false
    0 instructions correctly executed so far
```

**gfrun 对照输出：**

```
Total Block number = 21195
Total Inst number = 164205
Suaccelss to Reach the End of Benchmark! R2 = 0
```

## 四、根因分析（已定位到代码行，附插桩日志）

### 4.1 定位链（对 main@045c224 应用材料包 `instrumentation/bfu_zero_target_trace.patch` 后的日志）

```
[C:36177][BFU.NA]: redir_zero_tgt pc=0x1207e fb_va=0x1207a isCond=0 branchType=7(BLK_BR_RET) resolve_taken=1 fbid=43166 raw_resolve_tgt=0x0
[C:36177][BFU.NA]: fetchq_zero_va fbid_next=43207 va_prev=0x1207a tag=0x0 stid=0 first_after_redirect=1
[C:36180][BFU.NA]: bhc_miss_zero_pa PERMANENT-STALL fbid=43207 va=0x0 stid=0
[C:40185][DFX.NA]: Deadlock detected at thread 0, cycles=40186 ...
```

### 4.2 对应反汇编（`moe_dispatch_v2.elf.diss`）

```
000000000001207a <.LBB4_16>:
   1207a: 0800        C.BSTART.STD
   1207c: 40d53041    FRET.STK    [ra ~ s2], sp!, 256    <- pc=0x1207e 所在指令（RET 类，BranchType=BLK_BR_RET=7）
```

### 4.3 因果链（4 步）

1. **后端对 FRET.STK 的 resolveTarget=0**：C:36130-36131 TMA 报告 B195/B190 load/store 跨块冲突（`nuke_pair_diag`）触发 NUKE_FLUSH；恢复路径中，pc=0x1207c 的 `FRET.STK` 块（fbid=43166）被判定 mispredict 重定向，**IEX 经 `iex_brob_rslvblk` 总线给出的 `resolveTarget = 0`**（`TimingSim/frontend/bctrl/BROB.cpp:871` `resolveBlock()` 存入 `header->resolveTarget`，`BIFU.cpp:213` 透传为 `bfuInfo->resolve_tgt`）。

2. **前端用 0 作为重定向目标**：`TimingSim/frontend/bctrl/bfu/bfu_brq.cpp:236` `BRQ::ResolveHeader()` 中，**非条件分支（`IsCond()==false`，RET 属此类）不重算目标**，直接 `fb->redir_info.tgt = machineInst->bfuInfo->resolve_tgt;`（=0）；随后 `BFU::CreateNewInfoToFetchQ()`（`bfu.cpp:2668`）创建 **va=0 的取指 FB**（fbid=43207, `first_after_redirect=1`）。

3. **BHC 对 pa=0 静默永久 stall**：`TimingSim/frontend/bctrl/bfu/bfu_bhc.cpp` 中 `BHC::fetch()`：
   - 第 72 行先置位 `stall_fb = fb;`（"mark the current fb is causing cache stall"）；
   - 第 85 行 miss 入队处有守卫 `if (pa_cl != 0)` —— **pa=0 时既不入 missq 也不做任何处理**（无 assert、无错误注入、无超时），miss 永远不会被填充，`stall_fb` 永不释放。

4. **死锁**：BFU 取指停滞 → 无新 block 进入/提交 → BROB B199（BPC 0x11468, STD FALL）卡在 commitPtr 4000 周期 → 死锁检测中止（exit 134）。

### 4.4 问题定性（两层）

- **表层（明确的模型健壮性缺陷）**：`BHC::fetch()` 对 pa=0 的 miss 是"半成品"处理——作者显然知道 pa=0 特殊（专门加了 `if (pa_cl != 0)` 守卫避免发出非法 miss），但**没有为 pa=0 路径安排任何出口**（报错/放弃该 FB/重新取指均可），导致 `stall_fb` 无声永久置位。任何来源的异常目标地址（分支解析错误、寄存器读值错误等）一旦产生 va=0 取指，模型必然死锁且无任何诊断信息。
- **深层（待模型维护者确认）**：为什么 IEX 对该 `FRET.STK` 给出 `resolveTarget=0`？该指令返回目标应为 ra（栈上恢复），怀疑 NUKE_FLUSH 恢复路径中该块的 ra/寄存器依赖值未正确执行或前递。本 issue 附带的插桩 patch 可直接复现此现象（`raw_resolve_tgt=0x0`），可在此基础上继续排查执行单元侧。

## 五、材料包内容

```
moe_dispatch_gfsim_ret_zero_tgt/
├── ISSUE.md                                # 本文件
├── README.md                               # 材料包说明
├── repro.sh                                # 一键复现（clone@045c224 → build → gfsim 死锁 + gfrun R2=0 对照）
│                                           #   bash repro.sh trace  # 应用插桩 patch，打印 va=0 定位链
├── moe_dispatch_v2.elf                     # 触发死锁的 ELF（gfrun R2=0 版）
├── moe_dispatch_v2.elf.diss                # 反汇编（0x1207c FRET.STK 证据）
├── kernels/moe_dispatch_v2.hpp             # 算子源码
├── instrumentation/bfu_zero_target_trace.patch   # BFU 三处插桩（git apply 于 main@045c224）
└── evidence/
    ├── gfsim_deadlog_main_clean.log        # 干净基座死锁完整日志
    ├── gfrun_r2_0_main_045c224.log         # 同一 ELF gfrun R2=0 通过
    └── instrumentation_trace.log           # 插桩定位链日志（§4.1 的 4 行）
```

## 六、与相关 issue / PR 的关系

- **#344 / PR #356**（标量密集 kernel 死锁）：机制不同。#344 为 BROB 标量 store 未完成导致 commit 卡死，PR #356 通过 flush 时核销 STQ 条目修复。本 issue 根因在 **BFU/BHC 取指路径**（va=0 → pa=0 永久 stall）。已验证：main@045c224 cherry-pick PR #356 后本死锁依旧复现（时间线/死锁点完全一致）。
- **#348**（moe_dispatch gfrun R2=1 精度问题）：算子侧问题，已通过修正算子实现解决（修复后 gfrun R2=0，即本材料包 ELF）。本 issue 是该算子进入 gfsim 时序仿真后暴露的**模型侧**新问题，建议独立跟踪。
- 死锁点 `B199 [STD FALL] STD flushed` 表现与 #344 相似（均卡 commit），但根因不同——若无本插桩定位，极易误判为 #344 同类。

## 七、建议的修复方向（供参考）

1. **必须**：`BHC::fetch()` 中 pa=0 的 miss 分支增加显式处理——至少 `ASSERT`/`LOG_ERROR` + 放弃该 FB 并让上层重新取指，消除无声永久 stall；
2. **建议**：排查 IEX/NUKE 恢复路径对 `FRET.STK`（间接跳转/返回类）`resolveTarget` 的计算，确认为何为 0；
3. **建议**：`BRQ::ResolveHeader()` 对非条件分支 `resolve_tgt==0` 增加合法性检查（目标地址为 0 几乎必然是异常）。
