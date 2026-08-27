# Compiler workloads

This layer validates ordinary C/C++ compilation, ELF/link contracts, runtime
semantics, and auto-vectorization. It is deliberately separate from explicit
TileOP microbenchmarks and one-level operators.

The source bundle is pinned in `sources.lock.json`. Fetch it without vendoring
third-party source into this repository:

```bash
python3 benchmark/compiler-workloads/fetch_sources.py
```

The command prints the locked `workloads/` directory. Use that path to run the
smoke portfolio with the mandated main compiler checkout:

```bash
python3 benchmark/compiler-workloads/run_portfolio.py \
  --compile-only
```

`run_portfolio.py` defaults to the locked source path produced by
`fetch_sources.py`. Use `--workloads-root` only to override that location.

Add `--run-command '<runtime> {exe}'` for semantic execution. PolyBench is
enabled with `--polybench`; TSVC requires both `--tsvc` and `--qemu`. ctuning is
opt-in with `--ctuning-root` and `--ctuning-limit`.

Compile-only results are build evidence, not runtime correctness. Full PASS
requires the upstream workload runner's semantic markers/checksum checks.

Generated artifacts and consolidated JSON/Markdown reports are written under
`output/compiler-workloads/`.
