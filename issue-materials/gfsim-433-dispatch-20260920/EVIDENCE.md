# SuperScalarModel #433 附件材料：4-PE 动态 shape MoE 算子 gfsim 卡死——指令级定位与证据链

**日期**：2026-09-20　**提交人**：PR #163（SuperNPUBench）验证流程
**模型**：SuperScalarModel main @ `905ef557`（含 #694/#742/#745/#746）
**算子源**：SuperNPUBench PR #163 @ `208f032`（本分支即该 commit + 本材料目录）
**工具链**：llvm `4a3e0bdb`（clang 15.0.4，`linx64v5-unknown-linux-musl`）
**配置**：`--conf fourpe` 与 `-s core.threadCount=4 core.vec_core_num=4` 双配置，三例失败签名一致

> **重要**：本文档包含对本人 2026-09-20 早前两条评论中 moe_dispatch_mt_dyn 定性的**勘误**（§2.4）——深入地址级 trace 分析后，"跨 PE 陈旧读"结论被推翻，实际根因是 **T0（leader）前端在 tile-store 块后取指跑飞、barrier(2) flag 写从未执行**。

> ## ★ 最终结论（2026-09-20 定稿，详见 SuperScalarModel #765）
>
> **根因：前端在 0x117b2——4 字节 `BSTART.TLSU TSTORE`（0x117b0）的中间第 3 字节——起始建块**。被劈开的编码乱译出 `BSTART.STD DIRECT, 0x8207d2`（跳入栈区的垃圾直接分支），T0 于 C:15086 退休该垃圾块后**架构级跳入栈区**执行垃圾代码（零退休/零派发/零访存，BROB 空，fetch 停在 0x82c890，C:25025+ 对栈区发起指令 L2 读）；真实 TSTORE tile 命令因编码被劈开**从未形成**（全 trace 零生命周期事件），T0 永远到不了 barrier(2) flag 写 → slot 0 恒为 1（真实值）→ T1/2/3 在 .LBB2_41 互等至死锁检测（C:25108）。
>
> - 静态版同位置块边界干净（0x11780 = BSTART.TLSU 地址本身）→ tile 命令正常完成 → rc=0；
> - `mega_moe_sim_mt_dyn` 同病理（四线程块全卡 0x116ce = 4 字节 `cmp.eqi` 的中间字节）；
> - **算子侧已排除**：修复 `reinterpret_cast` NULL 基址（GMBase=0x0，`fix/moe-dispatch-dyn-cumsum-flag` 分支，gfrun 仍 R2=0）后死锁签名**逐周期不变**（C:25108 / verified blocks=1245）——根因不在 store 地址而在块边界劈开；
> - 归类：moe_dispatch_mt_dyn / mega_moe_sim_mt_dyn → **#765**（新）；group_token_vec_mt_dyn → **#433 形态 1**（跨 PE 标量可见性，维持）。
>
> §2.2/§2.3 时间线中"T0 取指跑飞"的机制由此定稿；§2.4 勘误第 3 点的候选 ①（前端下一取指目标损坏）即为 #765 的劈块乱译。

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
   - **【定稿】① 已实锤并细化**：块边界落在 0x117b2（BSTART.TLSU@0x117b0 的第 3 字节），乱译为 `BSTART.STD DIRECT, 0x8207d2`（跳栈区），见 §5.8-5.10 的提取命令；② 排除（算子修复后死锁逐周期不变，见 §7）。

### 2.5 时序敏感性（为何静态版/单配置版通过）

- 静态 `moe_dispatch_mt` 同模型 rc=0；`cfgA-only` 19,179 cycles rc=0（`logs/gfsim-diag/diag_cfgA_only.log`）；`cfgB-only` rc=0（`diag_cfgB_only.log`）；**仅 driver 双调用顺序执行触发**（`for (c=0; c<2; ++c)` 的循环结构差异改变代码布局与前端/流水时序）。
- gfrun 4-PE R2=0（cfgA+cfgB 双配置校验，`logs/gfrun/`）——非算子问题。

---

## 3. group_token_vec_mt_dyn：根因定稿——**T2 自身已提交的 tid 溢出 store 被丢失**（store loss，非跨 PE 陈旧读）

（`logs/trace/trace_gtvec_mt_dyn_excerpt.log`，全量 trace 4.5GB 摘录；符号 `sPhaseDone=0x14438`，T0/T1/T2/T3 槽 = 0x14438/3c/40/44；栈 sp：T0=0x8045130, T1=0x10055130, T2=0x18065130, T3=0x20075130）

地址级证据链：

1. **C:95-102（窗口 A）**：四线程 driver 入口的 tid 溢出 `sdi a1,[sp,40]@0x11348` 均执行并退休，数据正确：T0→0x8045158=0、T1→0x10055158=1、**T2→0x18065158=2**、T3→0x20075158=3（T2 的退休行：C:101 `retire entry 46 ... data=0x2 ... |sdi a1, sp, 40`——**架构级已提交**）；
2. **C:193（窗口 B）**：T2 的溢出写入 SCB1：`insert entry. blk_addr 0x18065140, addr 0x18065158, size 8, data 0x2. Combine`（合并进已有条目）；
3. **C:174119-174125（窗口 B/D）**：T2 的 `ldi [sp,40]@0x1174a` 发起到 0x18065158 的 load（L1 miss → L2），**返回 0x0**——距写入 174K 周期，**同 PE 的已提交 store 不可见**（T3 同款 Combine 插入却正常可见 3，T2 为孤立丢失事件）；
4. **C:174133/174848（窗口 C）**：T2 的 barrier(1) 写 `sw@0x1174e` 因槽位索引 a0=0 **写错槽位**：→ 0x14438（T0 槽）而非 0x14440（T2 自己的槽）；首次为推测执行（B248@C:174133），flush 后重放（B23@C:174848）仍读到 0、仍写错；
5. **对照（窗口 C）**：T0→0x14438=1 ✓、T1→0x1443c=1 ✓、T3→0x14444=1 ✓（C:174846/850/855）——只有 T2 错位；
6. **后果**：0x14440 恒为 0（从未被写）→ 四线程按序检查 slot0 ✓ → slot1 ✓ → **slot2 ✗ 卡死**在 `.LBB1_70@0x1176a`（`lwi@0x11770` 读 0x14440，全 trace 16,375 次 load 全部返回 0——**这是真实架构值**，非陈旧读）→ 持续提交自旋至超时（rc=124，死锁检测器不触发）。

**对 #433 形态 1 的精化**：gtvec 的挂起不是"跨 PE 已提交 store 读到旧值"，而是"**已提交 store 被整个丢失**（本例为同 PE 的栈溢出写，SCB `Combine` 插入后丢失，嫌疑在 combine/evict 排空路径）"。丢失的 store 对所有 PE（含写入者自己）都不可见——该机制同样能解释形态 1 的全部"跨 PE 不可见"表象。建议模型侧优先排查 SCB1 的 combine/evict 数据保持（C:100-193 窗口，T2 的 blk_addr 0x18065140 条目）。

`cfgA-only` 同样超时（308,661 retired，`logs/gfsim-diag/diag_gtvec_cfgA.log`）；静态版同挂 ⇒ 家族性问题。

## 4. mega_moe_sim_mt_dyn / 静态版

- **dyn**：C:223712 死锁检测（verified blocks=6698 / minsts=0）；**四线程 BROB 231 项全部为 BPC 0x116ce [STD DIRECT] allocated**——0x116ce 是 `0x116cc`（4 字节 `cmp.eqi a2, 0`）的**中间字节**，不存在合法块头，前端在指令中间反复建块且全部无法完成。上下文（`diss/…mega_moe_sim_mt_dyn.elf.diss`）：`0x11656/0x1167c` memset(workspace)、`0x1168e TRACE.begin`、`0x116a8-0x116b8` tiling 加载（`ld/lwui [s0, 0/8/16/24/32]`）、`0x116c0-0x116d6` 契约检查、`0x116da` 条件早退分支。
- **【trace 定稿】前驱序列与发散点**（`logs/trace/trace_mega_moe_sim_mt_dyn_excerpt.log`，全量 2.2GB trace 的关键窗口摘录；全量可重跑复现）：
  - STID2 发散前的块分配序列（C:213561-213629）：`B28 0x115f4 [STD COND]`（workspace 填充循环末次迭代）→ `0x11656 [STD CALL]` → memset(`0x12f38/40/44/48`) → `0x1167c [STD CALL]` → memset ×2 → `0x1168e [STD FALL]`（TRACE.begin）→ `0x11692 [STD COND]`（契约早退测试）→ `B40 0x116a6 [STD COND]`（tiling 加载块）；
  - **B41 = 0x116ce [STD DIRECT]**（C:213645, STID2 首个）——落在 `cmp.eqi@0x116cc` 的中间字节，乱译为 **`C.BSTART.STD DIRECT, 0x116ce`（自环）** → 取指在 0x116ce 无限建块（trace 中共 14,034 次 0x116ce 事件），BROB 堆积 231 项全部无法完成；
  - 四线程发散时刻：STID2 C:213645 / STID1 C:213648 / STID0 C:213655 / STID3 C:213666（SPMD 同代码相继到达同一切分点）；
  - 与 dispatch 例同病理：**块边界落在 4 字节指令的 +2 偏移处**（dispatch: BSTART.TLSU@0x117b0→劈于 0x117b2，乱译跳栈区；mega: cmp.eqi@0x116cc→劈于 0x116ce，乱译自环）。
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

# 5.8 【劈块实锤】T0 块边界序列：0x117a6→0x117b2（0x117b2 = BSTART.TLSU@0x117b0 的第 3 字节）
grep "full block retired" trace.log | grep -E "\-ST0 " | tail -6 | grep -oP "BPC:0x[0-9a-f]+ TPC:0x[0-9a-f]+"

# 5.9 【乱译内容】块 0x117b2 的逐条退休注记（BSTART.STD DIRECT 跳栈区 0x8207d2）
grep "retire entry" trace.log | grep "BPC:0x117b2" | grep -oP "\|.{0,60}"

# 5.10 【真实 TSTORE 从未形成】tile 命令零生命周期（对照同链 TEXPANDS/TSUB 均正常）
grep -cE "Wakeup by cmd.*0x117b0" trace.log          # = 0（真实 TSTORE）
grep -E "Wakeup by cmd.*STID0" trace.log | tail -3    # 最后事件 = C:15109 TSUB B218 retire + mapQ free

# 5.11 【静态对照】（logs/trace/trace_moe_dispatch_mt_STATIC_fourpe_t1.log.gz）
zcat trace_moe_dispatch_mt_STATIC_fourpe_t1.log.gz > static.log
grep "retire entry" static.log | grep "BPC:0x11776" | tail -2   # 块干净终止于 0x11780（BSTART.TLSU 地址本身）
grep "Wakeup by cmd" static.log | grep "0x11780" | head -1      # C:14540 TSTORE 正常唤醒（GMBase:0x0 亦完成）
# 注：静态版同款 reinterpret_cast 亦取 NULL 基址（GMBase=0x0），但块边界干净 → tile 命令正常完成 → rc=0
```

## 5a. group_token_vec_mt_dyn 分析命令（对全量 trace；摘录见 logs/trace/trace_gtvec_mt_dyn_excerpt.log）

```
# flag 写入史（注意 0x14440/0x14444 槽位——grep 模式须覆盖 0x1444x）
grep "store resolve" gtvec_trace.log | grep -E "ADDR 0x1443[89c]|ADDR 0x1444[04]" | \
  sed -E 's/.*TPC (0x[0-9a-f]+) B[0-9]+:G0:R[0-9]+:(STID[0-3]).*ADDR (0x[0-9a-f]+) TYPE addr DATA (0x[0-9a-f]+).*/\1 \2 \3=\4/'
# 预期：0x1136a/72=复位0×4线程；0x1174e: STID0→0x14438=1, STID1→0x1443c=1, STID2→0x14438=1(×2 错槽!), STID3→0x14444=1；【无任何 0x14440 写】

# T2 溢出写的完整生命周期（提交后丢失）
grep -E "0x18065158" gtvec_trace.log
# 预期：C:100 store resolve data=0x2 → C:193 SCB1 insert "Combine" → C:174119/174839 load 返回 0x0

# T2 的 ldi [sp,40] 返回值
grep "TPC:0x1174a" gtvec_trace.log | grep "\-ST2 " | grep "completed" | grep -oP "dst0:a0-P\([0-9]+\) data=0x[0-9a-f]+"

# 自旋读 0x14440 的地址与值
grep "Send req to lsu: TPC 0x11770" gtvec_trace.log | grep -oP "ADDR 0x[0-9a-f]+" | sort | uniq -c
grep "TPC:0x11774" gtvec_trace.log | grep -c "data=0x0,src1"
```

## 5b. mega_moe_sim_mt_dyn 分析命令（摘录见 logs/trace/trace_mega_moe_sim_mt_dyn_excerpt.log）

```
# 发散前驱（STID2）：B28 0x115f4(填充循环) → memset×2 → 0x1168e → 0x11692 → B40 0x116a6(tiling加载) → B41 0x116ce(劈块!)
awk -F'[][]' '{n=split($2,c,":"); cyc=c[2]+0; if (cyc>=213556 && cyc<=213646) print}' mega_trace.log | grep -E "Alloc BROB Entry.*STID2"
# 首个 0x116ce 块的乱译内容：C.BSTART.STD DIRECT, 0x116ce（自环）
grep -m 3 "BPC:0x116ce" mega_trace.log
# 各线程发散时刻：STID2 C:213645 / STID1 C:213648 / STID0 C:213655 / STID3 C:213666
for st in 0 1 2 3; do grep -E "Alloc BROB Entry.*STID$st BPC 0x116ce" mega_trace.log | head -1; done
```

## 6. 材料清单

```
elf/    8 个 ELF（4 dyn + 4 静态对照）
diss/   对应 8 份 llvm-objdump -dl 反汇编
fixed-build/          算子修复版（reinterpret_cast→正确构造，分支 fix/moe-dispatch-dyn-cumsum-flag）：
                      ELF + diss + gfrun R2=0 日志 + 全量 gfsim trace（gz 16.8MB，证明修复后死锁签名逐周期不变）
logs/gfsim-fourpe/   8 份 gfsim --conf fourpe 日志（含死锁 dump）
logs/gfsim-smt4/     6 份 gfsim -s core.threadCount=4 core.vec_core_num=4 日志
logs/gfsim-diag/     cfgA-only / cfgB-only / gtvec-cfgA-only 诊断构建日志
logs/gfrun/          4 份 gfrun（-t 1）日志尾部（R2=0 证据）
logs/trace/
  trace_moe_dispatch_mt_dyn_fourpe_t1.log.gz        dispatch dyn 全量 trace（16.8MB）
  trace_moe_dispatch_mt_dyn_excerpt.log             dispatch 关键窗口摘录（2MB）
  trace_moe_dispatch_mt_STATIC_fourpe_t1.log.gz     静态对照全量 trace（12.9MB，边界干净→PASS 的决定性对照）
  trace_mega_moe_sim_mt_dyn_excerpt.log             mega 发散窗口摘录（前驱序列 B28-B41 + 首个 0x116ce 块）
  trace_gtvec_mt_dyn_excerpt.log                    gtvec 五窗口摘录（tid 溢出→SCB Combine→读 0→写错槽→自旋）
  （gtvec/mega 全量 trace 各 2.2-4.5GB 未打包，按 §5a/§5b 命令用 repro 重跑）
repro.sh             一键复现
```

## 7. 算子修复验证（fixed-build/）

`fix/moe-dispatch-dyn-cumsum-flag` 分支修复 `reinterpret_cast<gm_st*>(windowState+4)` 的 NULL 基址（TSTORE/TLOAD GMBase=0x0，实际写往地址 0 而非设计的 windowState+16）：

- gfrun 4-PE **R2=0**（`fixed-build/gfrun_fixed_R2_0.log`）——修复保持功能正确；
- gfsim `--conf fourpe` 死锁签名**逐周期不变**（C:25108 / verified blocks=1245；`fixed-build/trace_..._FIXED_fourpe_t1.log.gz`）——**证明 gfsim 挂起根因不在 store 地址**，而在 #765 的劈块（修复版 `ldi [a1,16]`→`addi a1,16` 等长替换不改变切分点 0x117b2）；
- 同款 wart 仍存在于 `moe_dispatch_mt.hpp`（静态）与 `moe_dispatch_v2.hpp`（未触发，留 PR 作者定夺）。
