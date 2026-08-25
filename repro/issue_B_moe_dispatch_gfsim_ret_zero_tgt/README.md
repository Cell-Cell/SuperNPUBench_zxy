# repro: gfsim deadlock on moe_dispatch (RET redirect resolveTarget=0)

SuperScalarModel **main @ `045c224`** 上 `moe_dispatch_v2` 的 gfsim 死锁复现材料包。
issue 正文见 `ISSUE.md`（可直接粘贴到 GitHub issue）。

## 一句话

`moe_dispatch_v2` ELF 在 **gfrun 功能模型上 R2=0 全过**（164,205 条指令、21,195 个 block），
但在 **gfsim 时序模型上必然死锁**（exit 134, `Deadlock detected ... verified blocks=12487`）。
根因：**`FRET.STK`（pc=0x1207c）重定向时后端给出的 `resolveTarget=0`**，前端随之创建
**va=0 的取指请求**，BHC 对 **pa=0 的 miss 静默跳过（不入 missq、无超时、无报错）**，
`stall_fb` 永久置位 → 取指停滞 → 死锁。

## 目录结构

```
.
├── ISSUE.md                                # issue 正文（现象/根因/定位/复现，可直接粘贴）
├── README.md                               # 本文件
├── repro.sh                                # 一键复现: clone@045c224 → build → gfsim 死锁 + gfrun R2=0 对照
│                                           #   bash repro.sh trace  # 额外应用插桩 patch 打印 va=0 定位链
├── moe_dispatch_v2.elf                     # 触发死锁的 ELF（gfrun R2=0 版）
├── moe_dispatch_v2.elf.diss                # 反汇编（含 0x1207c FRET.STK 证据）
├── kernels/
│   └── moe_dispatch_v2.hpp                 # 算子源码（PTO 移植版，窗口数据已标量化访问）
├── instrumentation/
│   └── bfu_zero_target_trace.patch         # BFU 三处插桩（基于 main@045c224，git apply 即可）
└── evidence/
    ├── gfsim_deadlog_main_clean.log        # 干净基座 gfsim 死锁完整日志（含 nuke_pair_diag）
    ├── gfrun_r2_0_main_045c224.log         # 同一 ELF gfrun R2=0 通过（main 基座）
    └── instrumentation_trace.log           # 插桩版定位链日志（4 行闭环）
```

## 快速复现（3 条命令）

```bash
git clone https://github.com/LinxISA/SuperScalarModel.git && cd SuperScalarModel
git checkout 045c224 && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make gfsim gfrun -j
./bin/gfsim -f /path/to/moe_dispatch_v2.elf    # 预期: Deadlock detected, exit 134
./bin/gfrun -t 1 -f /path/to/moe_dispatch_v2.elf   # 对照: R2 = 0 正常完成
```

或直接 `bash repro.sh`（自动完成 clone/build/复现/对照），`bash repro.sh trace` 额外输出插桩定位链。

## 定位链（插桩日志，evidence/instrumentation_trace.log）

```
[C:36177][BFU.NA]: redir_zero_tgt       pc=0x1207e fb_va=0x1207a isCond=0 branchType=7(BLK_BR_RET) resolve_taken=1 fbid=43166 raw_resolve_tgt=0x0
[C:36177][BFU.NA]: fetchq_zero_va       fbid_next=43207 va_prev=0x1207a tag=0x0 stid=0 first_after_redirect=1
[C:36180][BFU.NA]: bhc_miss_zero_pa PERMANENT-STALL fbid=43207 va=0x0 stid=0
[C:40185][DFX.NA]: Deadlock detected at thread 0, cycles=40186, waitCycles exceeded threshold, verified blocks=12487, verified minsts=0
```

对应反汇编（moe_dispatch_v2.elf.diss）：

```
000000000001207a <.LBB4_16>:
   1207a: 0800        C.BSTART.STD
   1207c: 40d53041    FRET.STK    [ra ~ s2], sp!, 256    <- pc=0x1207e 所在 RET 指令
```

## 版本信息

| 组件 | 版本 |
|---|---|
| SuperScalarModel（死锁主体） | `main` @ `045c224`（Merge PR #321） |
| 算子 | `moe_dispatch_v2`（SuperNPUBench `ops-20260823` 配对工具链编译） |
| 对照 | 同一 ELF：gfrun `R2 = 0`（aarch64 Linux, gcc 9 / clang 15 工具链） |
