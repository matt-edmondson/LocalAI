# Per-GPU VRAM-Aware Scheduling & Multi-GPU Model Placement

**Status:** Design approved for implementation planning — 2026-06-03
**Author:** Matt Edmondson (with Claude)
**Scope:** Distributed (cluster) mode scheduling and worker backend placement.

## 1. Background & root cause

**Incident:** GPU workers stuck with VRAM in use but the orchestrator reporting "no
backend loaded", and new backends failing to load.

Confirmed against the live `localai` namespace and the code:

- Workers heartbeat **aggregate** free VRAM, summed across all GPUs
  (`xsysinfo.GetGPUAggregateInfo` → `Config.heartbeatBody`, `core/services/worker/registration.go:131`).
- The scheduler treats each node as a single VRAM pool
  (`backend_nodes.available_vram`, `NodeRegistry.FindNodeWithVRAM`).
- Nothing assigns a backend to a specific GPU. The worker spawns processes with
  `process.WithEnvironment(os.Environ()...)` (no per-backend `CUDA_VISIBLE_DEVICES`,
  `pkg/model/process.go:161`), and the diffusers backend defaults to `cuda:0`
  (`backend/python/diffusers/backend.py:621`, `device_map=None`).
- `SmartRouter.estimateModelVRAM` (`core/services/nodes/router.go:906`) returns **0**
  for non-GGUF models (diffusers/SDXL, arbitrary safetensors). With `estimatedVRAM == 0`
  the scheduler **skips the VRAM filter and the soft reservation** (`router.go:735`,
  `router.go:876`) and falls back to VRAM-blind `FindIdleNode`/`FindLeastLoadedNode` —
  which the code's own comment warns "will happily pick a node that's about to OOM."

**Observed on `noctis-2gpu` (2× RTX A2000 12 GB):** one SDXL model filled GPU0
(11 751 / 12 282 MiB) while GPU1 sat at 4 MiB used (≈11.9 GB free). The scheduler,
seeing ≈12 GB aggregate free, kept placing more diffusers backends on the node; each
defaulted to `cuda:0` → could not allocate on the full GPU0 → never passed the 7-minute
readiness probe → killed → retried → orchestrator eventually gave up
(`Failed to install backend via NATS`).

The VRAM holder is a **live, idle, loaded** SDXL backend (`State: S`), not a leaked or
hung process. The three prior fixes (`e689d16b`, `a9ecdf41`, PR #6) only tuned reconciler
reap *timing* — a layer above the actual defect.

## 2. Goals / non-goals

**Goals**
- Spread independent single-GPU backends across all GPUs on a node (stop piling on `cuda:0`).
- Support splitting one large model across multiple GPUs **on the same node**.
- Reason about VRAM **per-GPU**, not per-node-aggregate.
- Provide usable VRAM estimates for models that cannot be parsed from metadata.
- Replace retry-forever / silent give-up with bounded retries and a clear terminal error.

**Non-goals**
- GPUs as first-class schedulable registry units / full schema refactor (a possible future
  evolution; see §8).
- Multi-node sharding (a single model spanning GPUs across different hosts).
- A new backend-side VRAM-reporting protocol (backends stay unchanged; the diffusers
  multi-GPU path via `device_map="auto"` already exists).

## 3. Chosen approach

**Approach 1 — per-GPU heartbeat + worker-side pinning, scheduler decides the GPU set.**
Per-GPU visibility is plumbed to the registry so node-selection can reason per-GPU; the
**scheduler** decides the node and the exact physical GPU set; the **worker enforces** it by
setting `CUDA_VISIBLE_DEVICES` on the backend process. Multi-GPU is **automatic from the VRAM
estimate**; estimates come from a **file-size heuristic seed corrected by a measured value**
cached in the registry.

Decisions locked during design:
- Separate `node_gpus` table (not a JSON column on `backend_nodes`); per-GPU `reserved_vram`.
- `CUDA_VISIBLE_DEVICES` pinning is uniform across **all** backend types.
- Measured estimates cached in the registry DB, keyed by model.
- **Best-fit** packing (smallest free GPU that still fits — preserve whole GPUs for big models).
- **Fewest-GPUs** spanning.

## 4. Data foundation

### 4.1 Per-GPU VRAM in heartbeat & registry
- `Config.heartbeatBody()` keeps sending aggregate `available_vram` (UI/back-compat) and
  **adds** `gpus: [{index, total_vram, free_vram, used_vram}]`, built from the already-collected
  `[]xsysinfo.GPUMemoryInfo` (per-GPU detection exists today; only the aggregate is currently surfaced).
- `HeartbeatUpdate` gains `GPUs []GPUInfo`.
- New registry model **`NodeGPU`** → table `node_gpus`:
  `node_id, gpu_index, total_vram, free_vram, reserved_vram, updated_at`
  (one row per physical GPU; PK `(node_id, gpu_index)`).
- The soft reservation moves to **per-GPU** granularity (`node_gpus.reserved_vram`), reusing
  the existing reserve/clear-on-heartbeat/rollback-on-failure pattern. Node-level
  `available_vram` remains as a derived/compat value.
- A heartbeat that omits `gpus[]` (GPU-detection blip) leaves the last-good `node_gpus` rows
  in place under a staleness guard; the scheduler falls back to node-level aggregate for that
  node (today's behavior) until per-GPU data returns.

### 4.2 Measured-estimate store
- New registry model **`ModelVRAMEstimate`** → table `model_vram_estimates`:
  `model_name, backend, vram_bytes, source ('heuristic'|'measured'), gpu_count_observed, updated_at`.
  Cluster-shared, survives restarts.
- New `vram.EstimateHeuristic(files)` (in `pkg/vram`): sum of on-disk weight sizes × dtype
  factor + a fixed overhead constant (text encoders / VAE / CUDA context). Replaces the
  "return 0" path for unparseable models and is the cold-start seed.
- **Estimate lookup order** at placement: `measured` row → existing GGUF/HF metadata estimate
  (still wins when the model *can* be parsed) → `heuristic`.

## 5. Placement & enforcement

### 5.1 Placement (scheduler — `SmartRouter.scheduleNewModel`)
1. Compute `estimate` via §4.2 lookup (now effectively always > 0, so the VRAM-blind fallback
   that caused the incident is removed).
2. For each candidate node, read its per-GPU free list (free − per-GPU reserved) and pick the GPU set:
   - **Fits one GPU** (some GPU's effective free ≥ estimate) → `K = 1`, choose the **smallest
     free GPU that still fits** (best-fit).
   - **Needs spanning** (no single GPU fits, but the largest `K` GPUs' free sum ≥ estimate) →
     assign the **fewest** GPUs that fit; `K` = that count.
   - **Doesn't fit** → next node → existing eviction path, re-checking per-GPU after each evict.
3. **Per-GPU soft reservation**: reserve ≈ `estimate / K` on each assigned GPU
   (`node_gpus.reserved_vram`); cleared on the node's next heartbeat (worker is source of truth),
   rolled back explicitly on load failure.
4. Thread the chosen **physical GPU indices** and `K` forward; record them on the `NodeModel` row.

### 5.2 Enforcement (worker + ModelOptions)
- `messaging.BackendInstallRequest` and the `InstallBackend(...)` sender gain `GPUIndices []int`.
- `backendSupervisor.startBackend` → `ModelLoader.StartProcess` accept an env override and set
  **`CUDA_VISIBLE_DEVICES=<comma-joined indices>`** on the spawned process (overriding the
  inherited value). No assignment (CPU / no GPU) → unset, preserving today's behavior.
- When the frontend builds `ModelOptions` for LoadModel (`core/backend/options.go`), it derives
  the multi-GPU knob from `K`:
  - diffusers / vLLM → `TensorParallelSize = K` (`K > 1` ⇒ existing `device_map="auto"`; `K = 1`
    ⇒ single device).
  - llama.cpp → `TensorSplit` even across `K` (`MainGPU` relative to the visible set).
- Because `CUDA_VISIBLE_DEVICES` already restricts the process to its assigned cards, the backend
  re-numbers them `0..K-1`; `TensorParallelSize=K` / `TensorSplit`-over-`K` is internally
  consistent and no physical index leaks into backend code.

## 6. Closing the loop: measurement, cold-start, errors

### 6.1 Measurement (heuristic → measured)
Per-PID attribution is unreliable inside a container (the driver reports host PIDs; the
container's `/proc` is namespaced → `nvidia-smi` shows `[Not Found]`). Use a **namespace-proof
per-GPU delta** instead:
- The frontend records each assigned GPU's `used` baseline just before `backend.install`.
- After a **successful LoadModel**, it reads the updated per-GPU `used` from the node's heartbeat
  data (polled 1–2 intervals to settle); the delta on the assigned GPUs is the model's footprint
  → written as a `measured` row.
- Record only when **cleanly attributable** (no other concurrent install on those GPUs during the
  window); otherwise skip and keep the heuristic, logging the skip. One clean observation suffices
  and thereafter wins over the heuristic.

### 6.2 Cold-start / OOM fallback (removes the retry-forever loop)
- First load of an unmeasured model uses the heuristic seed.
- On failure (OOM / readiness timeout), **bump that model's cached estimate** (backoff factor or
  a recorded "failed-at" floor) so the next attempt reserves more / spans more GPUs instead of
  retrying identically.
- Always **release per-GPU reservations on failure**, and **cap retries** with a clear terminal
  error (e.g. *"model needs more VRAM than any GPU set on any node can provide"*) rather than the
  current silent give-up.
- Over-estimation (heuristic too high) wastes capacity only until the first measured observation
  corrects it.

### 6.3 Edge cases
- GPU-detection blip → node-level aggregate fallback with staleness guard (see §4.1).
- External / non-LocalAI GPU usage is already reflected in real `free_vram`, so best-fit avoids it.
- CPU / RAM-only workers unchanged (no `gpus[]` → existing RAM path).
- Reaper / unload paths (`reconciler.go`, `deleteProcess`, remote unload) must release per-GPU
  reservations and clear the `NodeModel` GPU assignment.

## 7. Component / file map (indicative)

| Area | File(s) | Change |
|------|---------|--------|
| Per-GPU detection accessor | `pkg/xsysinfo/gpu.go` | expose per-GPU `[]GPUMemoryInfo` (aggregate already sums it) |
| Heartbeat body | `core/services/worker/registration.go` | add `gpus[]` to `heartbeatBody()` |
| Heartbeat ingest | `core/http/endpoints/localai/*heartbeat*`, `core/services/nodes/registry.go` | accept `gpus[]`; upsert `node_gpus` |
| Registry models | `core/services/nodes/registry.go` | `NodeGPU`, `ModelVRAMEstimate`; per-GPU reserve/release; per-GPU fit queries |
| Heuristic estimator | `pkg/vram/` (new) | `EstimateHeuristic(files)` |
| Estimate lookup | `core/services/nodes/router.go` | `estimateModelVRAM` new lookup order |
| Placement | `core/services/nodes/router.go` | `scheduleNewModel` per-GPU fit / span / best-fit; bounded retries + terminal error |
| Install message | `core/services/messaging/*`, NATS `InstallBackend` sender | add `GPUIndices []int` |
| Process start | `core/services/worker/supervisor.go`, `pkg/model/process.go` | env override → `CUDA_VISIBLE_DEVICES` |
| ModelOptions | `core/backend/options.go` | derive `TensorParallelSize`/`TensorSplit` from `K` |
| Replica row | `core/services/nodes/registry.go` (`NodeModel`) | record assigned GPU indices + `K` |
| Measurement | `core/services/nodes/router.go` | post-LoadModel per-GPU delta → `measured` row |

## 8. Migration / rollout
- Two additive GORM tables (`node_gpus`, `model_vram_estimates`); no destructive migration.
  Existing node-level columns retained for compatibility.
- Mixed-version safe: a worker that doesn't send `gpus[]` is handled by the aggregate fallback;
  an install request without `GPUIndices` behaves as today (no pinning).
- Single-GPU nodes: `K` is always 1, every GPU set is `[0]`; behavior is equivalent to today
  but now backed by a real (non-zero) estimate.

## 9. Testing
- **Unit:** heuristic sizing (files → bytes); placement table (fits-one-GPU best-fit /
  spans-fewest / doesn't-fit→evict) over per-GPU-free scenarios; per-GPU reserve/release
  atomicity; `K` → `TensorParallelSize`/`TensorSplit` derivation; `CUDA_VISIBLE_DEVICES`
  injection from `GPUIndices`.
- **Registry integration (ginkgo, à la `registry_test.go`):** `gpus[]` heartbeat → `node_gpus`
  rows; per-GPU fit query; measured-estimate write/read; reservation cleared on heartbeat.
- **Regression (the incident):** node with one full GPU + one free GPU → new model lands on the
  free GPU, no OOM/readiness-timeout loop; a >single-GPU model spans both and loads via
  `device_map="auto"`.

## 10. Future evolution (out of scope now)
- Promote GPUs to first-class schedulable units (Approach 2) if fragmentation / multi-node
  spanning become requirements.
- Backend-reported VRAM (authoritative footprint from the backend over gRPC) to replace the
  measured-delta heuristic.
