# Kill Abandoned Backend Loads — Design

**Date:** 2026-06-03
**Branch:** `fix/kill-abandoned-backend-loads` (from `origin/custom`)
**Status:** Approved

## Problem

When the frontend's `LoadModel` gRPC call to a remote worker backend fails — timeout,
caller disconnect, transport error, or a backend-reported failure — `scheduleAndLoad`
(`core/services/nodes/router.go`) just returns the error. The worker-side backend
process, already spawned by `installBackendOnNode` *before* staging and `LoadModel`,
keeps running and keeps loading. It completes invisibly, holds RAM/VRAM, and is
unkillable by the control plane: no `NodeModel` registry row was ever created
(`SetNodeModel` runs only after a successful load), so the stale-replica reaper and
per-GPU VRAM accounting are both blind to it.

### Incident (2026-06-03)

- Two concurrent SDXL cold loads on noctis hit the then-5m `LoadModel` timeout; the
  frontend gave up but both Python processes kept loading → stacked RAM → worker pod
  OOMKilled (24Gi limit, exit 137).
- An abandoned animagine-xl-4.0 load completed *after* the deadline and held 7.1GB on
  GPU0 with the load recorded as failed → no replica row → reaper and per-GPU
  accounting blind → next placement on that GPU would CUDA-OOM.
- PR #9 (15m timeout) makes timeouts rarer but widens the orphan window to 15m.

## Goal

When a load is abandoned after the worker process was spawned, kill that exact
worker-side process and release the per-GPU soft VRAM reservations — synchronously,
before the load-coalescing advisory lock is released.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Stop semantics | **Synchronous acked stop** | Requests for the same model queue on the `model-load:<model>` advisory lock. Fire-and-forget would let a queued request re-install while the kill is in flight: the worker's "already running" fast path hands it the doomed process's address, then the kill lands and its `LoadModel` dies. Acked stop inside the lock closes the window. |
| Crash coverage | **Targeted fix only** | The error path covers timeout, disconnect, staging failure, and backend-reported failure — the actual incident. Frontend-crash-mid-load needs an orphan-process reaper (new protocol surface); deferred as follow-up. |
| Reservation release | **Include, via `placement` struct refactor** | Worker heartbeats reset `reserved_vram` to 0 every interval (registry.go — worker is source of truth), so release after a multi-minute failure is a guarded no-op. It still matters for fast failures inside the first heartbeat window, and keeps accounting symmetric with the existing install-failure rollback. |
| Wire protocol | **Upgrade existing `backend.stop` subject `Subscribe` → `SubscribeReply`** | The messaging layer already no-ops replies when `msg.Reply == ""` (messaging/client.go:93), so old frontends publishing to the same subject keep working. A new subject (`backend.stop.sync`) would add a second handler + fallback re-publish for no behavioral gain. Health-polling instead of an ack was rejected: a busy mid-load process fails HealthCheck without being dead. |

## Design

### Data flow

`scheduleAndLoad` arms a deferred cleanup after `scheduleNewModel` succeeds (process
spawned, reservations held) and disarms it after a successful `LoadModel`:

```
scheduleNewModel ok → arm cleanup
  staging fails        ─┐
  LoadModel errors      ├─→ cleanupAbandonedLoad(node, model, replica, reservedGPUs, perGPUReserve)
  res.Success == false ─┘      1. StopBackendAndWait(node.ID, "model#replica")  ← exact processKey, 30s ack timeout
  LoadModel ok → disarm        2. ReleaseVRAMOnGPU for each reserved GPU (guarded no-op if heartbeat reset)
```

- Cleanup runs on a fresh `context.Background()`-derived context: the request ctx is
  cancelled in the disconnect case — that's the point.
- Cleanup runs **inside the advisory lock** (it's part of `scheduleAndLoad`, which
  `Route` calls under `WithLockCtx`), so a queued request for the same model cannot
  re-install until the process is confirmed dead.
- Cleanup is best-effort: every step logs on failure; the original load error is
  always what's returned to the caller.

### Component changes

1. **`core/services/nodes/router.go`**
   - Refactor `scheduleNewModel`'s five return values into a `placement` struct:
     `{node, addr, replicaIndex, gpuSet, reservedGPUs, perGPUReserve}`.
   - Add `cleanupAbandonedLoad(node, trackingKey, replicaIdx, reservedGPUs, perGPUReserve)`.
   - Arm/disarm in `scheduleAndLoad` covering the three post-install failure points.

2. **`core/services/nodes/unloader.go`**
   - Add `StopBackendAndWait(nodeID, backend string) error` to
     `RemoteUnloaderAdapter` and the `NodeCommandSender` interface. 30s
     request-reply (`RequestJSON`), same wire payload as today's fire-and-forget
     `StopBackend` (`{"backend": "<key>"}`). Timeout / no-responders → warn +
     return error; caller logs and continues.
   - Existing `StopBackend` (Publish) untouched for eviction/unload callers.

3. **`core/services/worker/lifecycle.go`**
   - `backend.stop` subscription: `Subscribe` → `SubscribeReply`.
   - Handler body unchanged, but wrapped in a goroutine (matching
     `handleBackendInstall`) so a slow stop doesn't head-of-line-block the
     subscription; replies `BackendStopReply{Success: true}` after
     `stopBackend`/`stopAllBackends` returns.

4. **`core/services/worker/supervisor.go`**
   - Bound `stopBackendExact`'s best-effort `Free()` call to ~10s
     (currently `context.Background()`, unbounded) so a wedged mid-load backend
     can't stall the kill or the ack. `proc.Stop()` always runs regardless.

5. **`core/services/messaging`**
   - Add `BackendStopReply{Success bool, Error string}`. Request wire format
     unchanged.

### Safety / edge cases

- **Exact processKey only.** The kill targets `trackingKey#replicaIdx`. A bare model
  ID prefix-matches in the worker's `resolveProcessKeys` and would kill legitimate
  sibling replicas (e.g. the reconciler's `M#1`). Locked in with a dedicated test.
- **Coalescing guard.** The advisory lock serializes Route-vs-Route, but the
  reconciler's `ScheduleAndLoadModel` calls `scheduleAndLoad` without it. If a racing
  load of the same (node, model, replica) *succeeded* — a `NodeModel` row exists in
  state `loaded` for that exact triple — skip the kill; still release our
  reservations (the release is underflow-guarded and heartbeat-reset anyway).
- **Mixed-version cluster.** New frontend + old worker: the stop executes via the old
  plain-Subscribe handler but never acks → 30s timeout, logged, original load error
  returned (i.e. degrades to today's fire-and-forget semantics). Old frontend + new
  worker: publish has no reply subject → reply is a guarded no-op.
- **`bumpEstimateOnLoadFailure`** keeps firing on these paths (pre-existing
  behavior, unchanged).

### Out of scope

- **Frontend crash mid-load** — no error path runs; needs the deferred
  orphan-process reaper (worker/frontend reconciling running processes against
  registry rows with a grace period > load timeout).
- **RAM stacking during concurrent load windows** — two loads both still inside
  their timeout can still stack RAM; that's the readiness-timeout / capacity
  follow-up (worker `LOCALAI_BACKEND_READINESS_TIMEOUT` bump), not this change.
- **`SetNodeModel` failure after a successful load** — healthy process with a
  missing row; pre-existing, different bug.

## Testing (TDD)

Frontend — `core/services/nodes/abandoned_load_cleanup_test.go`, reusing
`fakeUnloader` + `fakeBackendClientFactory`:

1. `LoadModel` error → stop sent with exact key `model#replica` to the right node,
   reservations released, original error returned.
2. `LoadModel` returns `!Success` → same.
3. Staging failure → stop sent, `LoadModel` never attempted.
4. Successful load → **no** stop call.
5. Abandoning `M#0` never stops `M#1`.
6. Loaded registry row exists for the exact (node, model, replica) triple → kill
   skipped, reservations still released.
7. `StopBackendAndWait` times out (old worker) → cleanup logs, original load error
   still propagates.

Worker — `core/services/worker/`:

- `handleBackendStop` replies success after stopping (via existing fake-NATS seams).
- `Free()` bound verified if an injectable seam exists; otherwise accepted as
  untestable-without-refactor and noted in the plan.

Known pre-existing failures: 11 Windows path-separator failures in
`staging_keys_test.go` (green on Linux) — unrelated.

## Deployment

Merge to `custom` → manual image build
(`gh workflow run "fork: build + push to GHCR" --ref custom -f tag=custom`, ~10m) →
Keel auto-rolls the cluster. Mixed-version windows during rollout degrade gracefully
(see Safety).
