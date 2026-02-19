# Online Learning Implementation Spec

Status: Draft
Owner: lfg.cpp core
Last updated: 2026-02-17

## Purpose

This spec defines how to implement online learning in `lfg.cpp` using:

1. Signal-driven memory routing (entropy/surprise/confidence).
2. In-engine vector cache for retrieve/store.
3. QLoRA-style adaptation (frozen quantized base, train LoRA adapters only).

It translates `docs/online-learning.md` from architecture vision into implementable steps.

## Current State (as of 2026-02-17)

1. Signal monitors are implemented and exposed via session APIs.
2. Retrieval/store policy is caller-orchestrated today.
3. `lfg_generate_result.n_retrievals` is currently reserved/always `0`.
4. LoRA adapter load/apply is implemented for inference.
5. Training APIs (`lfg_opt_init`, `lfg_opt_epoch`) are present.
6. Current training path marks F32 model tensors trainable; quantized base tensors are not train targets.
7. Adapter API documents a lifecycle constraint: adapters should be loaded before context creation.

## Key Decisions

1. Use QLoRA approach.
2. Keep quantized base model frozen during online learning.
3. Train only LoRA A/B parameters in higher precision.
4. Add an in-engine tiered vector cache (2/3 layers) as the default memory backend.
5. Keep policy controllable: safe defaults in-engine, optional external override.
6. Require both inference-signal improvements and downstream outcome quality for adapter promotion.
7. Online training must be interruptible by inference, with inference as highest priority.

## Non-Goals

1. Full-parameter fine-tuning of quantized base weights.
2. Distributed training in first implementation.
3. Full vector DB backend integration in core (local in-memory cache first).

## Requirements

### Functional

1. Store surprise/confidence events into vector cache with metadata.
2. Retrieve on entropy events, rewind to checkpoint, inject top-K memory snippets, continue generation.
3. Count and report retrievals in generate results.
4. Persist training examples as `(input, retrieval_context, output, outcome_label)` tuples.
5. Trigger QLoRA training jobs from accumulated examples.
6. Support adapter promotion and rollback.

### Safety/Quality

1. Cap retrieval loops to prevent rewind cascades.
2. Gate promotion on quality metrics, not inference signals alone.
3. Preserve existing zero-alloc hot path guarantees where possible.
4. Inference must preempt training under load without user-visible stalls.
5. Enforce throughput guardrails with automatic degradation modes.

## Architecture

### A) In-Engine Vector Cache

Add a session-owned fixed-cap memory store:

1. Flat contiguous embedding array (`capacity x n_embd`).
2. Parallel metadata arrays (source, timestamp, token span, signal type, text reference).
3. Similarity search by dot product (L2-normalized vectors).
4. Eviction policy: ring overwrite (MVP), optional score/age policy later.

### A.1) Memory Tiering (2/3 Layers)

Default topology:

1. `L1 hot` (episodic/recent):
   1. High write/read frequency.
   2. Short retention.
   3. Supports rapid rewind-time retrieval.
2. `L2 warm` (semantic/merged):
   1. Deduped or merged representations.
   2. Medium retention and lower write churn.
   3. Primary layer for repeated domain recall.
3. `L3 cold` (optional archive):
   1. Long retention and compressed summaries.
   2. Lower-priority retrieval unless high similarity.
   3. Can be disabled for 2-layer deployments.

### B) Signal Routing Loop

1. Surprise high on ingestion: store input memory candidate.
2. Confidence span event: store output memory candidate.
3. Entropy event: retrieve top-K candidates.
4. If retrieval above threshold and within budget: rewind + inject + resume.
5. Increment `n_retrievals` in `lfg_generate_result`.

### C) Online Training Data Pipeline

1. Record each retrieval-augmented episode.
2. On successful outcome label, append training example.
3. Build mini-datasets periodically (time- or count-based).

### D) QLoRA Training + Adapter Lifecycle

1. Extend training path to target adapter A/B tensors (not base quantized weights).
2. Train candidate adapter from recent dataset window.
3. Evaluate candidate on:
   1. Inference metrics: surprise/retrieval/decode-failure trends.
   2. Outcome quality metrics: task success/rework/error rates.
4. Promote if both improve or hold.
5. Roll back automatically on degradation.
6. Support explicit "bad adapter" flagging with blacklist and cooldown.

### D.1) Preemptible Training Scheduler

1. Training runs at lower priority than inference by default.
2. Training executes in small resumable slices (micro-batch/step checkpoints).
3. Before each training slice, scheduler checks inference pressure.
4. On pressure signal, pause training immediately and release compute resources.
5. Resume training from saved optimizer/state checkpoint when pressure clears.
6. Adapter promotion is disallowed while training is paused or in-flight.

### D.2) Adapter Activation Gating

1. Maintain a domain centroid (or k-centroids) computed from the adapter's training window embeddings.
2. At inference, compute query embedding and compare against centroid(s).
3. Enable adapter only if similarity >= activation threshold.
4. Provide a hard override to disable all adapters under guardrail pressure.

### D.3) Bad Adapter Flagging

1. Provide a manual and automatic way to mark an adapter as "bad".
2. Manual: explicit blacklist API call.
3. Automatic: flag if metrics regress beyond threshold for a sustained window.
4. Bad adapters are immediately disabled and excluded from future promotion.
5. Record reason, timestamp, and evidence metrics for auditability.

## API Additions (Proposed)

1. Vector cache configuration:
   1. capacity
   2. max_text_bytes
   3. retrieval_top_k
   4. retrieval_min_score
   5. max_retrievals_per_generate
2. Vector cache operations:
   1. enable/disable
   2. clear
   3. inspect stats
3. Generate result fields:
   1. `n_retrievals` (implemented)
   2. `n_memory_writes` (optional)
4. Training control:
   1. online training enable/disable
   2. dataset window and trigger thresholds
   3. candidate adapter status/query
   4. promote/rollback hooks
   5. pause/resume/cancel training
   6. training priority mode (background, opportunistic)
   7. training checkpoint interval (preemption granularity)
   8. adapter activation threshold(s) and centroid mode
   9. blacklist/ban adapter API

## Performance and ETA Estimation

There is no universal LoRA training duration. ETA is measured on target hardware.

Recommended estimator:

1. Run a short calibration train job (2-5 minutes).
2. Measure `train_tokens_per_second`.
3. Estimate:
`total_time_sec ~= (total_train_tokens * epochs / train_tokens_per_second) * overhead_factor`
4. Use `overhead_factor = 1.1` to `1.4`.

Example sanity checks:

1. `100k` tokens at `500 tok/s` ~= `200s` per epoch.
2. `1M` tokens at `500 tok/s` ~= `2000s` (~33 min) per epoch.

## Throughput Guardrails

Guardrails are hard limits that keep serving performance stable. They are enforced
automatically; online learning degrades or pauses when limits are exceeded.

Default targets:

1. `p50_tokens_per_second` regression <= 5%.
2. `p95_latency` regression <= 10%.
3. Per-session memory cap enforced for cache + training queues.

Guarded behaviors:

1. Retrieval is capped per generation (`max_retrievals_per_generate`, cooldown).
2. L1-only retrieval in hot path; L2/L3 lookup is async.
3. Writes are queued; dedupe/merge happens off the hot path.
4. Training runs at lower priority and is preemptible.

Degradation ladder (auto):

1. Disable `L3` retrieval.
2. Disable `L2` retrieval.
3. Raise retrieval threshold / reduce top-k.
4. Disable retrieval; keep writes.
5. Disable writes; keep inference-only.
6. Pause training entirely.

## Implementation Phases

### Phase 1: Memory MVP

1. Add session vector cache.
2. Wire surprise/confidence writes and entropy retrieves.
3. Populate `n_retrievals`.
4. Add tests for retrieve/store behavior and retrieval budget.

### Phase 2: Training Example Capture

1. Add episode recorder for `(input, retrieval, output, label)`.
2. Add local persistence format and replay loader.
3. Add metrics dashboards/counters.

### Phase 3: QLoRA Training Integration

1. Add adapter-parameter training path.
2. Add `bench_qlora_step` for throughput and ETA calibration.
3. Add candidate adapter training/eval lifecycle.

### Phase 4: Promotion + Rollback

1. Add promotion gate logic (signals + outcome quality).
2. Add atomic adapter switch strategy compatible with context lifecycle constraints.
3. Add rollback and health checks.

## Validation and Acceptance Criteria

1. Entropy-triggered retrieval changes output when relevant memory exists.
2. Retrieval counts are non-zero and correctly reported.
3. Surprise/confidence writes populate cache with valid embeddings.
4. QLoRA candidate training runs end-to-end on quantized-base deployment.
5. Promotion and rollback are automated and auditable.
6. Online loop lowers retrieval rate over repeated domain exposure without harming outcome quality.
7. Under induced inference load, training pauses quickly, inference latency remains within SLO, and training resumes without losing progress.

## LoRA Effectiveness Metrics

Primary metrics to evaluate candidate adapters:

1. `retrieval_rate`: retrievals per generated token or per task.
2. `near_duplicate_write_rate`: new writes with cosine similarity above duplicate threshold to existing cache entries.
3. `train_overlap_write_rate`: new writes similar to cache items that were part of the adapter's training window.
4. `train_overlap_retrieval_rate`: retrieval hits similar to cache items from the adapter's training window.
5. `outcome_quality`: task success, correction/rework rate, and factual error rate (app-defined).

Interpretation:

1. Decreasing retrieval and duplicate-like storage is a positive sign.
2. These are necessary but not sufficient promotion signals.
3. If retrieval/write overlap rates drop while outcome quality degrades, treat as confident wrongness and reject/rollback.

Bad adapter criteria (example defaults):

1. `outcome_quality` degrades > X% for Y consecutive windows.
2. `retrieval_rate` drops while error/rework rises.
3. Human override signals "bad" for the adapter.

## Open Questions

1. Adapter hot-swap strategy:
   1. Recreate contexts on promotion, or
   2. Add runtime-safe adapter switching API.
2. Outcome label source:
   1. Explicit application callback, or
   2. Heuristic proxy fallback.
3. Persistence backend for memory/training store:
   1. In-memory only (MVP), or
   2. Optional disk-backed store.
4. Preemption granularity defaults:
   1. Time-based slice budget, or
   2. Micro-batch-count checkpoints.
