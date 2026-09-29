#!/usr/bin/env python3
# SL2 viz trace 分析: 统计 prior 请求流 (证明"每 cycle 1 条永续流"饿死 tile 派发)
# 用法: python3 analyze_preq_stream.py <sl2viz_file> [freeze_cycle]
import re, sys
pat_preq = re.compile(r'"c":(\d+)[^}]*"k":"PREQ"[^}]*"inst":(\d+)[^}]*"addr":(\d+)')
pat_inst = re.compile(r'"c":(\d+)[^}]*"k":"INST"[^}]*"inst":(\d+)[^}]*"addr":(\d+)')
path = sys.argv[1]
freeze = int(sys.argv[2]) if len(sys.argv) > 2 else 0
buck, addrs, insts = {}, {}, []
with open(path, errors='replace') as f:
    for line in f:
        m = pat_preq.search(line)
        if m:
            c = int(m.group(1))
            b = c // 10000
            buck[b] = buck.get(b, 0) + 1
            if c > freeze:
                a = int(m.group(3))
                addrs[a] = addrs.get(a, 0) + 1
        m = pat_inst.search(line)
        if m and int(m.group(1)) > freeze:
            insts.append((int(m.group(1)), m.group(2), int(m.group(3))))
print("PREQ per 10K cycles (恒定 10000 = 每 cycle 1 条永续流):")
for b in sorted(buck):
    print(f"  {b*10000}: {buck[b]}")
print(f"distinct addr post-{freeze}: {len(addrs)}")
for a, n in sorted(addrs.items(), key=lambda x: -x[1])[:6]:
    print(f"  0x{a:X} x{n}")
print(f"INST (tile) tokens post-{freeze}: {len(insts)}  (期望 = 卡死前出生一次后归零)")
for c, i, a in insts[:4]:
    print(f"  c={c} inst={i} addr=0x{a:X}")
