# SuperScalarModel #433 附件材料：4-PE 动态 shape MoE 算子 gfsim 卡死——指令级定位与证据链

**日期**：2026-09-20　**提交人**：PR #163（SuperNPUBench）验证流程
**模型**：SuperScalarModel main @ `905ef557`（含 #694/#742/#745/#746）
**算子源**：SuperNPUBench PR #163 @ `208f032`（本分支即该 commit + 本材料目录）
**工具链**：llvm `4a3e0bdb`（clang 15.0.4，`linx64v5-unknown-linux-musl`）
**配置**：`--conf fourpe` 与 `-s core.threadCount=4 core.vec_core_num=4` 双配置，三例失败签名一致

> **重要**：本文档包含对本人 2026-09-20 早前两条评论中 moe_dispatch_mt_dyn 定性的**勘误**（§2.4）——深入地址级 trace 分析后，"跨 PE 陈旧读"结论被推翻，实际根因是 **T0（leader）前端在 tile-store 块后取指跑飞、barrier(2) flag 写从未执行**。

---

## 1. 复现（repro.sh）

| ELF | gfrun 4-PE | gfsim fourpe | gfsim SMT4 |
|---|---|---|---|
| moe_combine_mt_dyn | R2=0 | **PASS** | **PASS** |
| moe_dispatch_mt_dyn | R2=0 | **FAIL**（rc=134，C:25108 死锁检测） | FAIL（同签名） |
| group_token_vec_mt_dyn | R2=0 | **FAIL**（rc=124 超时自旋） | FAIL（同签名） |
| mega_moe_sim_mt_dyn | R2=0 | **FAIL**（rc=134，C:223712） | FAIL（同签名） |
| moe_dispatch_mt / moe_combine_mt（静态对照） | R2=0 | PASS / PASS | PASS / PASS |
| group_token_vec_mt / mega_moe_sim_mt（静态对照） | R2=0 | FAIL / FAIL | — |

---

## 2. moe_dispatch_mt_dyn：cfgA 首次调用卡死的完整证据链（核心）

### 2.1 内存布局与关键 PC（`diss/solution_moe_dispatch_moe_dispatch_mt_dyn.elf.diss` + `llvm-nm`）

符号地址：`sMtPhaseDone = 0x15000`（槽位：T0=0x15000, T1=0x15004, T2=0x15008, T3=0x1500c）；`windowState = 0x2c000`；`predBuf = 0x27000`。

| PC | 指令/块 | 语义 |
|---|---|---|
| `0x1148c-0x11490` | `B.HINT TRACE.begin` + `C.BSTART.STD DIRECT 0x115de`（setret 0x1152c） | driver 调用 kernel（cfgA/cfgB 各一次） |
| `0x115de` | `FENTRY [ra~s8], sp!, 256` | `moe_dispatch_mt_dyn` 入口 |
| `0x11682` | `call dispatch_pack_mt_dyn`（0x11dc6，setret 0x11692） | Phase-1 pack（tile 写 windowData/windowFlag） |
| `0x11692-0x1169e` | `addtpc 4 / subi / c.movi 1 / sw [t#2, s4<<2]` | **mtBarrier(1) flag=1 写**（T1/2/3 路径） |
| `0x116a2-0x116ba` | `.LBB2_8/9/10`：`lwi.u + c.setc.eq zero` | barrier(1) 检查（`flag==0` 自旋） |
| `0x11716-0x11724` | `sw 0x15000 = 1` | T0 路径的 barrier(1) 写（实测 C:13846） |
| `0x11758` | `call cal_cumsum_mt_dyn`（0x11f18，setret 0x11772） | Phase-2 cumsum |
| `0x120aa` | `FRET.STK [ra~s8], sp!, 256` | **cumsum 返回**（四线程均正常退休） |
| `0x11772-0x11786` | `ldi` 链 + `BSTART.TEPL TEXPANDS` | 返回路径 → PE0-only `write_cumsum_flag` |
| `0x117b0-0x117be` | `BSTART.TLSU TSTORE FP32`（512B）→ windowState+4 | write_cumsum_flag 落盘（T0 最后退休的块边界 0x117b2→0x117c2） |
| **`0x117c2-0x117de`** | **`BSTART.TLSU TLOAD FP32`（512B）← windowState+4** | **check_cumsum_flag 回读——T0 冻结点，从未执行** |
| `0x11804-0x11826` | `.LBB2_37`：`c.movi 2 / sw t#1, [t#2, a2<<2]` | **mtBarrier(2) flag=2 写**（T0 的这次从未执行） |
| `0x1182a-0x1185e` | `.LBB2_41/42/43/44`：`lwi.u [t#1, 2004/1994/1984/…]` + `setc.ltui t#1, 2` | barrier(2) 检查；**.LBB2_41 的 lwi@0x11830 实测读 0x15000（T0 槽）** |
| `0x11a58-0x11a82` | `setc.ltui t#1, 3` ×4 | barrier(3) 检查 |
| `0x1156e-0x11598` | `setc.ltui t#1, 4` ×4 | driver 侧 mtBarrier(4) 检查（**全 trace 0 次执行** ⇒ 死锁在 cfgA kernel 内部） |

### 2.2 时间线（`logs/trace/`，`-t 1` 全量 trace）

| 周期 | 事件 | 证据（§5 命令） |
|---|---|---|
| C:96-106 | driver 复位：4×STID 写 0x15000-0x1500c = 0 | store resolve TPC 0x113f8/0x11400/0x11404/0x11408 |
| C:13846 | T0 barrier(1) 写：`0x15000 = 1` | store resolve TPC 0x11724 |
| ~C:13-14K | T1/2/3 barrier(1) 写（TPC 0x1169e）；**四线程全部通过 barrier(1)** | store resolve TPC 0x1169e ×3 |
| C:14547-14627 | phase-1 pack 的 TLSU tile 写仍在落地（GMBase 0x1f00-0x25e00，BPC 0x11e80/0x11ede） | `Wakeup by cmd` |
| C:14585 / 14618 / 14633 | T1/2/3 FRET 返回后执行 barrier(2) 写：`0x15004/0x15008/0x1500c = 2` | store resolve TPC 0x11826（STID1/2/3） |
| C:14587 / 14623 / 14635 | T1/2/3 的 0x11812 写块退休 | `full block retired BPC:0x11812`（ST1/2/3） |
| C:14596-25107 | **T1/2/3 在 0x1182a 自旋且持续提交（各 2086 次，共 6258）**；`.LBB2_41` 的 `lwi@0x11830` 全部 6263 次读 **ADDR 0x15000** | `full block retired BPC:0x1182a`；`Send req to lsu: TPC 0x11830` |
| C:15065 | T0 的 FRET（0x120aa）退休 | `full block retired`（ST0） |
| C:15071-15086 | T0 执行 PE0-only `write_cumsum_flag` 链并全部退休：0x11772→0x1177c→0x11786(TEXPANDS)→0x11798(TSUB)→0x117a6→**0x117b2(TSTORE 512B→windowState+4)** | `full block retired`（ST0 tail） |
| **C:15086 之后** | **T0 再无任何块退休**。下一块 `0x117c2`（check_cumsum_flag 的 512B TLOAD 回读 windowState+4）从未进入执行；死锁时 T0 的 BROB 为 **空**，fetch 停在**栈地址 0x82c890/0x82c8d0**（把栈数据当指令取指：L2 ReadReq 0x82c800/0x82c840/0x82c880 @ C:25025-25029） | dump：`stall_fb stid0 addr 0x82c890`；`EMPTY BROB ENTRY thread: 0` |
| （从未发生） | **T0 的 barrier(2) 写（`0x15000 = 2`）从未执行——全 trace 无任何来源写 0x15000=2** | store resolve 全表（§5.2） |
| C:25108 | `Deadlock detected`（无提交线程 = T0） | `logs/gfsim-fourpe/…dispatch_mt_dyn.log` 尾部 dump |

### 2.3 因果链

```
T0 在 write_cumsum_flag 的 512B TSTORE（0x117b0）块退休后（C:15086）
  └─> 下一块 0x117c2（同地址 512B TLOAD 回读）从未被取指/执行
        —— T0 前端下一取指目标损坏：fetch 停在栈地址 0x82c8xx
  └─> T0 永远到不了它的 barrier(2) flag 写（0x11826 → 0x15000 = 2）
        —— slot 0 保持 1（barrier(1) 值，C:13846 写入）
  └─> 四线程的 barrier(2) 首检 .LBB2_41（lwi@0x11830 读 0x15000）永远读到 1
        —— 这是真实架构值，不是陈旧读！
  └─> T1/2/3 自旋提交 2086×3 次至 C:25107；T0 零提交
  └─> C:25108 线程级死锁检测触发（T0）
```

### 2.4 对本人早前两条评论的勘误

1. **"37,332 次陈旧读（写入已提交但读到 1）"——撤销**。该计数的构成：T1/2/3 的 6,258 次真实自旋检查 + T0 的 ~31,357 次**推测执行**检查（T0 的 fetch-ahead 越过冻结点后预取到 0x1182a 自旋块并推测执行）。读到的 1 是 slot 0 的真实架构值（T0 从未写 2）。跨 PE 可见性在本例**工作正常**：barrier(1) 四线程全过、T1/2/3 的 flag=2 写已提交且各自的检查能读到彼此。
2. **"T1/2/3 自旋块提交停滞"——撤销**。T1/2/3 持续提交至死锁瞬间；卡住的是 **T0**（leader）。
3. 修正后的主嫌疑（两个候选成分，黑盒无法进一步区分，先后未知）：
   - **① 前端下一取指目标损坏**：tile-store 块（0x117b2）退休后，T0 的取指未去 fall-through 0x117c2 而停在栈地址——与 mega_moe_sim_mt_dyn 的"四线程块卡指令中间地址 0x116ce"同族（共享预测/取指结构损坏）；
   - **② 同 PE TSTORE→TLOAD 512B tile 回读依赖停滞**（0x117b0 存 → 0x117cc 读同地址）：#576（tile 依赖等待停滞）家族。
   - 注：T0 的 BROB 为空 + 0x117c2 从未分配 ⇒ 更直接支持 ①；但 15086→25025 之间 T0 取指的具体行为（未发 L2 读）留待模型侧确认。

### 2.5 时序敏感性（为何静态版/单配置版通过）

- 静态 `moe_dispatch_mt` 同模型 rc=0；`cfgA-only` 19,179 cycles rc=0（`logs/gfsim-diag/diag_cfgA_only.log`）；`cfgB-only` rc=0（`diag_cfgB_only.log`）；**仅 driver 双调用顺序执行触发**（`for (c=0; c<2; ++c)` 的循环结构差异改变代码布局与前端/流水时序）。
- gfrun 4-PE R2=0（cfgA+cfgB 双配置校验，`logs/gfrun/`）——非算子问题。

---

## 3. group_token_vec_mt_dyn：形态 1（跨 PE 标量可见性）维持

- 四线程全部自旋在 `0x1176a / .LBB1_70`：`lwi.u [t#1, -812]` + `c.setc.eq t#1, zero`（barrier(1) 的 `flag==0` 检查）；SMT4 配置同签名（0x1175e）。
- **与 dispatch 的本质区别**：四线程均在持续提交自旋块（886,705 retired @1.2M cycles，rc=124 超时，死锁检测器不触发）——自旋块在提交 ⇒ 各自的 flag 写块已提交 ⇒ 读到 0 是**已提交 store 的不可见** = 形态 1。
- `cfgA-only` 同样超时（308,661 retired，`logs/gfsim-diag/diag_gtvec_cfgA.log`）⇒ 首次调用 barrier(1) 即不可见，排除 PR 双调用复位竞态；静态版同挂 ⇒ 家族性问题。

## 4. mega_moe_sim_mt_dyn / 静态版

- **dyn**：C:223712 死锁检测（verified blocks=6698 / minsts=0）；**四线程 BROB 231 项全部为 BPC 0x116ce [STD DIRECT] allocated**——0x116ce 是 `0x116cc`（4 字节 `cmp.eqi a2, 0`）的**中间字节**，不存在合法块头，前端在指令中间反复建块且全部无法完成。上下文（`diss/…mega_moe_sim_mt_dyn.elf.diss`）：`0x11656/0x1167c` memset(workspace)、`0x1168e TRACE.begin`、`0x116a8-0x116b8` tiling 加载（`ld/lwui [s0, 0/8/16/24/32]`）、`0x116c0-0x116d6` 契约检查、`0x116da` 条件早退分支。
- **静态**：C:618044 死锁检测，四线程各卡 `0x12740 / 0x1266c / 0x12926 / 0x124ea`；`0x12926` = `lwi.u [t#1,-8] + setc.ltui t#1, 2`（**mtBarrier(2) 检查**）；`0x12740/0x1266c` = `lw [t#3, t#1]` 索引轮询（跨 PE 数据到达等待）。卡点全为标量指令 ⇒ 非 #576；日志无 nuke/flush（dyn 0 条、static 2 条）⇒ 非 #681 机制 ⇒ 归形态 1 家族。

---

## 5. 分析命令复现（对 `logs/trace/trace_moe_dispatch_mt_dyn_fourpe_t1.log.gz`）

```bash
zcat trace_moe_dispatch_mt_dyn_fourpe_t1.log.gz > trace.log

# 5.1 flag 数组全部写入史（TPC/STID/ADDR/DATA）
grep "store resolve" trace.log | grep -E "ADDR 0x1500[0-9c] " | \
  sed -E 's/.*TPC (0x[0-9a-f]+) B[0-9]+:G0:R[0-9]+:(STID[0-3]).*ADDR (0x[0-9a-f]+) TYPE addr DATA (0x[0-9a-f]+).*/\1 \2 \3=\4/' | sort | uniq -c
# 预期：0x113f8/40/04/08=复位0；0x1169e STID1/2/3=1；0x11724 STID0=1；0x11826 STID1/2/3=2；【无任何 0x15000=2】

# 5.2 .LBB2_41 检查块的读地址（全部 6263 次读 0x15000）
grep "Send req to lsu: TPC 0x11830" trace.log | grep -oP "ADDR 0x[0-9a-f]+" | sort | uniq -c

# 5.3 各线程最后退休块（ST# 为线程号；T0 最后 = 0x117b2 块 @C:15086）
for t in 0 1 2 3; do grep "full block retired" trace.log | grep -E "\-ST$t " | tail -3; done

# 5.4 T1/2/3 的 barrier(2) 写块与自旋块退休（ST1/2/3 各 2086 次自旋提交）
grep "full block retired" trace.log | grep "BPC:0x11812"   # 3 条：C:14587/14623/14635（ST1/2/3）
grep "full block retired" trace.log | grep "BPC:0x1182a" | grep -oP "\-ST[0-9]" | sort | uniq -c

# 5.5 T0 前端跑飞（栈取指）
grep -E "0x82c8[0-9a-f]{2}" trace.log | grep -E "ReadReq|stall_fb" | head
# 死锁 dump（BROB 头部/stall_fb/TLSU windows）见 logs/gfsim-fourpe/new_solution_moe_dispatch_moe_dispatch_mt_dyn.log 尾部

# 5.6 driver mtBarrier(4) 检查从未执行（⇒ 死锁在 cfgA kernel 内）
grep -c "BPC:0x1156e" trace.log   # = 0

# 5.7 pack tile 写晚于 barrier(1) 落地
grep "Wakeup by cmd" trace.log | grep -E "C:145[4-9][0-9]|C:146[0-2][0-9]"
```

## 6. 材料清单

```
elf/    8 个 ELF（4 dyn + 4 静态对照）
diss/   对应 8 份 llvm-objdump -dl 反汇编
logs/gfsim-fourpe/   8 份 gfsim --conf fourpe 日志（含死锁 dump）
logs/gfsim-smt4/     6 份 gfsim -s core.threadCount=4 core.vec_core_num=4 日志
logs/gfsim-diag/     cfgA-only / cfgB-only / gtvec-cfgA-only 诊断构建日志
logs/gfrun/          4 份 gfrun（-t 1）日志尾部（R2=0 证据）
logs/trace/          dispatch dyn 全量 trace（gzip 16.8MB）+ 关键周期窗口摘录（2MB）
repro.sh             一键复现
```
