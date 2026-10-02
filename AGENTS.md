# AGENTS.md — SimQ

You are the sole maintainer. Own it: relentlessly make the project better every session — find bugs, close issues, raise coverage, improve docs/perf, keep CI green. Don't wait to be asked; investigate, fix, verify, commit, push.

Rust quantum SDK. Workspace: `simq-core` (Circuit/Gate trait) → `simq-gates` (concrete gates) → `simq-state` (statevectors) → `simq-compiler` (passes/decomposition) → `simq-sim` (Simulator) → `simq-backend` (Transpiler/Router/backends) → `simq` (facade: `QuantumCircuit` fluent API, re-exports all as `simq::{core,gates,state,sim,compiler,backend}`). Plus `simq-macros`, `simq-py` (pyo3 bindings).

## Maintainer loop (every session)

- Check `git status`, open GitHub issues (`gh issue list`), and failing/skipped tests; pick the highest-severity correctness bug first.
- Reproduce before fixing (regression test in the right `tests/` dir), fix honestly: **error loudly over silent wrong answers** (empty gate lists with fidelity 1.0, `Ok(vec![])` zero-SWAP claims, and lossy approximations are the recurring failure mode).
- Verify with scoped runs, then the full CI order below; update docs/tests alongside the fix.
- Commit focused changes and push (feature branch, never force-push, never amend others' commits). Leave the tree green.

## Commands (CI order matters)

```
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-features --workspace --exclude simq-py
cargo test --all-features --workspace --exclude simq-py
cargo test --doc --workspace --exclude simq-py
```

- `simq-py` is **always excluded** from cargo build/test: pyo3 `extension-module` only links under Python. Test it via maturin + pytest in `simq-py/`.
- Coverage gate: tarpaulin `--fail-under 97.5` (see `ci.yml`; excludes `simq-py`, `simq-sim/src/gpu.rs`, `simq-macros`, `ibm_quantum.rs`, gate cache, `simd/*`).
- Profiles: dev `opt-level=1`, test `opt-level=2` — tests build slowly; prefer scoped runs.

## Scoped testing (do this)

- `cargo test -p <crate> --lib -q` or `cargo test -p <crate> --test <name> -q`.
- Never bare `cargo test -p <crate> --tests`: it compiles every file in `tests/`, including broken untracked repros (e.g. `simq-backend/tests/sampling_bias.rs` fails on missing `rand` import and has no assertions). Untracked `*_bug.rs` / `reset_bug.rs` / `write_bytes_*.rs` are scratch, not repo tests.
- `cargo test -p simq-core --lib` currently fails pre-existing (`serialization`/`bincode` cfg errors) — verified on clean tree, don't chase it.
- Pre-existing failures (verified on clean tree, unrelated to in-flight work): `simq-sim --lib batch_eval::...shares_the_fusion_cache_across_instances` and `simq-sim --test comprehensive_e2e fusion_cache_hits_across_repeated_same_shape_runs` (both expect fusion-cache hits, get 0).
- Stable `cargo fmt` prints warnings for nightly-only `rustfmt.toml` keys (`wrap_comments`, `imports_granularity`, etc.) — harmless. Run `cargo fmt -p <crate>` after editing.

## Gotchas that bite

- `DecompositionResult.gates: Vec<Arc<dyn Gate>>` carries **gate types only, no qubit indices**. `TwoQubitGateInstruction::CNOT{control,target}` / `MultiQubitInstruction` direction is lost — replay positionally against the instruction order (SWAP = 3×`CNot` where order encodes direction). Documented on the struct; don't "fix" by substituting a different gate.
- `can_decompose()` is a **shape check only** (qubit count + matrix present). `decompose()` still `Err`s for generic 2q unitaries (magic-basis unimplemented), non-π/4 Clifford+T angles, and √iSWAP (no gate type in `simq-gates` — error, never substitute `ISwap`). Never `unwrap()` after `can_decompose()`.
- `SabreRouter::route()` intentionally **returns `Err`** (no circuit input, can't plan). Real routing is `Router::find_swap_chain` (greedy shortest-path, used by `Transpiler::map_and_route`). `heuristic_score` is test-only until circuit-aware SABRE lands.
- `optimize_gate_sequence` / `TwoQubitDecomposer::optimize_decomposition` / Medium-Heavy transpiler levels are documented pass-throughs — don't treat as bugs.
- `simq-py` audit ignores `RUSTSEC-2025-0020`, `RUSTSEC-2026-0177` (pyo3 0.21, tracked separately).

## Hygiene

- Don't commit scratch: `.day_of_year`, `.modules`, `nowa_trade_automation.py` (contains an embedded private key — never commit), or any untracked `*_bug*.rs`.
