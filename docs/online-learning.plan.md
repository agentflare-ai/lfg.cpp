# Online Learning Implementation Plan

Status: Draft
Owner: lfg.cpp core
Last updated: 2026-02-17

This plan executes `docs/online-learning.spec.md` as phased checklists.
Each phase begins and ends with explicit "gates green" checks.

## Phase 0: Baseline and Guardrails

- [ ] Gate: Green — Baseline tests/benchmarks pass; no new regressions.
- [ ] Define throughput SLO targets and measurement methodology.
- [ ] Add guardrail metrics collection (p50 tok/s, p95 latency, memory cap).
- [ ] Add guardrail degradation ladder controller (no-op stubs are fine).
- [ ] Add smoke tests for guardrail trigger order.
- [ ] Gate: Green — Guardrails measurable and no regressions vs baseline.

## Phase 1: Tiered Vector Cache (L1/L2/L3)

- [ ] Gate: Green — Guardrails still green before cache integration.
- [ ] Implement L1 hot cache (fixed-cap ring, dot-product search).
- [ ] Implement L2 warm cache (dedupe/merge, async fill).
- [ ] Implement optional L3 cold cache (archive, lowest priority).
- [ ] Add metadata + counters (writes, reads, overlap rates).
- [ ] Wire cache config knobs (capacity, top-k, min score, TTLs).
- [ ] Unit tests for insert/evict/search/dedupe by tier.
- [ ] Gate: Green — Cache tests pass, guardrails still green.

## Phase 2: Signal Routing Integration

- [ ] Gate: Green — Cache tests green before routing integration.
- [ ] Wire surprise/confidence events to cache writes.
- [ ] Wire entropy events to retrieval + rewind + inject.
- [ ] Enforce retrieval budgets and cooldowns.
- [ ] Populate `n_retrievals` and optional `n_memory_writes`.
- [ ] Integration tests for retrieval budget and correct counters.
- [ ] Gate: Green — Routing tests pass, guardrails still green.

## Phase 3: Training Example Capture

- [ ] Gate: Green — Routing stable before dataset capture.
- [ ] Define episode format `(input, retrieval, output, label)`.
- [ ] Implement rolling dataset window and persistence hooks.
- [ ] Add overlap tracking between new writes and training window.
- [ ] Add tests for dataset assembly and window rollover.
- [ ] Gate: Green — Dataset tests pass, guardrails still green.

## Phase 4: QLoRA Training Integration

- [ ] Gate: Green — Dataset capture stable before training.
- [ ] Add adapter-parameter training path (LoRA A/B tensors only).
- [ ] Implement training scheduler with micro-batch checkpoints.
- [ ] Implement inference-preemptible training control (pause/resume/cancel).
- [ ] Add `bench_qlora_step` for throughput calibration.
- [ ] Add unit test for preemption and resume correctness.
- [ ] Gate: Green — Training tests pass, guardrails still green.

## Phase 5: Promotion + Rollback

- [ ] Gate: Green — Training stable before promotion logic.
- [ ] Define promotion gates from spec (signals + outcome quality).
- [ ] Implement candidate adapter eval and promotion decisioning.
- [ ] Implement rollback path with audit logging.
- [ ] Add integration tests for promote/rollback edge cases.
- [ ] Gate: Green — Promotion tests pass, guardrails still green.

## Phase 6: End-to-End Validation

- [ ] Gate: Green — All component tests green.
- [ ] Run E2E scenario: repeated in-domain prompts.
- [ ] Verify retrieval/write overlap rates decrease.
- [ ] Verify outcome quality holds or improves.
- [ ] Verify guardrails degrade correctly under load.
- [ ] Gate: Green — E2E success and no throughput regressions.
