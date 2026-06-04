# Kill Abandoned Backend Loads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a remote model load is abandoned after the worker process was spawned (staging error, LoadModel timeout/disconnect/error, or backend-reported failure), synchronously kill the exact worker-side backend process and roll back per-GPU VRAM reservations.

**Architecture:** The frontend's `scheduleAndLoad` arms a deferred cleanup after `scheduleNewModel` succeeds and disarms it after a successful load. Cleanup sends an **acked** `backend.stop` (the existing subject upgraded from `Subscribe` to `SubscribeReply` — backward compatible because the messaging layer no-ops replies for plain publishes) targeting the exact `modelID#replica` process key, then releases the per-GPU soft reservations. Cleanup runs inside the `model-load:<model>` advisory lock so coalesced requests can't inherit a doomed process.

**Tech Stack:** Go, Ginkgo/Gomega tests, NATS request-reply via `core/services/messaging`, GORM registry.

**Spec:** `docs/superpowers/specs/2026-06-03-kill-abandoned-backend-loads-design.md`

**Branch:** `fix/kill-abandoned-backend-loads` (already created from `origin/custom`)

**Known pre-existing failures (NOT yours):** 11 Windows path-separator failures in `core/services/nodes/staging_keys_test.go` (green on Linux). If you see exactly those, ignore them.

**Commit style:** Conventional commits (`fix(nodes): ...`, `test(worker): ...`). No version tags, no Co-Authored-By lines.

---

### Task 1: Messaging wire types for acked backend.stop

The request payload `{"backend": "..."}` already rides `backend.stop`; it just has no shared type (the adapter uses an inline anonymous struct, and `core/services/nodes/unloader.go:18-21` has a dead private `backendStopRequest`). Add public types so the frontend and worker share one definition, plus a reply type.

**Files:**
- Modify: `core/services/messaging/subjects.go` (next to `func SubjectNodeBackendStop` — search for it)
- Modify: `core/services/nodes/unloader.go:18-21` (delete dead type), `core/services/nodes/unloader.go:253-262` (use the shared type)

- [ ] **Step 1: Add the wire types to subjects.go**

Directly below the `SubjectNodeBackendStop` function in `core/services/messaging/subjects.go`, add:

```go
// BackendStopRequest is the payload for a backend.stop NATS message. The
// same payload serves both delivery modes: fire-and-forget Publish (eviction,
// admin unload) and request-reply (the frontend's abandoned-load cleanup,
// which must hold the model-load advisory lock until the process is dead).
type BackendStopRequest struct {
	Backend string `json:"backend"`
}

// BackendStopReply is the worker's ack for a request-reply backend.stop.
// Fire-and-forget publishers never see it; old workers (plain Subscribe)
// never send it and the requester times out — degrading to fire-and-forget.
type BackendStopReply struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}
```

- [ ] **Step 2: Use the shared type in the adapter and delete the dead one**

In `core/services/nodes/unloader.go`, delete the dead private type at the top of the file:

```go
// backendStopRequest is the request payload for backend.stop (fire-and-forget).
type backendStopRequest struct {
	Backend string `json:"backend"`
}
```

And in `StopBackend` (line ~253), replace the inline anonymous struct:

```go
	req := struct {
		Backend string `json:"backend"`
	}{Backend: backend}
	return a.nats.Publish(subject, req)
```

with:

```go
	return a.nats.Publish(subject, messaging.BackendStopRequest{Backend: backend})
```

- [ ] **Step 3: Verify wire compatibility via the existing tests**

Run: `go test ./core/services/nodes/ -count=1 -args -ginkgo.focus="StopBackend"`
Expected: PASS — the existing `"with backend name publishes JSON"` spec (`unloader_test.go:181`) unmarshals the payload and asserts `payload.Backend`, proving the wire format is unchanged.

- [ ] **Step 4: Commit**

```bash
git add core/services/messaging/subjects.go core/services/nodes/unloader.go
git commit -m "refactor(messaging): shared BackendStopRequest/Reply wire types"
```

---

### Task 2: Worker — acked backend.stop handler

Upgrade the worker's `backend.stop` subscription from `Subscribe` to `SubscribeReply` so request-reply callers get an ack after the process is dead. The messaging layer's reply closure already guards `msg.Reply != ""` (`core/services/messaging/client.go:93`), so plain publishes from old frontends keep working.

**Files:**
- Create: `core/services/worker/lifecycle_stop_test.go`
- Modify: `core/services/worker/lifecycle.go:24` (subscription), `core/services/worker/lifecycle.go:104-118` (handler)

- [ ] **Step 1: Write the failing test**

Create `core/services/worker/lifecycle_stop_test.go`:

```go
package worker

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/mudler/LocalAI/core/services/messaging"
)

var _ = Describe("handleBackendStop", func() {
	// The handler runs its body in a goroutine (so a slow stop can't
	// head-of-line-block the subscription); the reply lands asynchronously.
	expectAck := func(payload []byte) {
		s := &backendSupervisor{processes: map[string]*backendProcess{}}
		replies := make(chan []byte, 1)
		s.handleBackendStop(payload, func(data []byte) { replies <- data })

		var raw []byte
		Eventually(replies, "2s").Should(Receive(&raw))
		var reply messaging.BackendStopReply
		Expect(json.Unmarshal(raw, &reply)).To(Succeed())
		Expect(reply.Success).To(BeTrue())
	}

	It("acks after stopping a specific process key", func() {
		// No process under this key — stopBackend no-ops, but the ack must
		// still arrive: the frontend's abandoned-load cleanup blocks on it.
		expectAck([]byte(`{"backend":"my-model#0"}`))
	})

	It("acks the stop-all form (empty payload)", func() {
		expectAck(nil)
	})
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./core/services/worker/ -count=1 -args -ginkgo.focus="handleBackendStop"`
Expected: COMPILE FAILURE — `s.handleBackendStop` has signature `func([]byte)`, the test passes two arguments. This is the TDD red.

- [ ] **Step 3: Implement the acked handler**

In `core/services/worker/lifecycle.go`, change the subscription (line 24):

```go
	s.nats.SubscribeReply(messaging.SubjectNodeBackendStop(s.nodeID), s.handleBackendStop)
```

Replace `handleBackendStop` (lines 104-118) with:

```go
// handleBackendStop is the NATS callback for backend.stop — stop a specific
// backend process. Fire-and-forget callers (eviction, admin unload) Publish
// and never read the reply; the frontend's abandoned-load cleanup uses
// request-reply and blocks on the ack so it can hold the model-load advisory
// lock until the process is confirmed dead. The body runs in a goroutine so
// a slow stop doesn't head-of-line-block other events on this subscription.
func (s *backendSupervisor) handleBackendStop(data []byte, reply func([]byte)) {
	go func() {
		var req messaging.BackendStopRequest
		if json.Unmarshal(data, &req) == nil && req.Backend != "" {
			xlog.Info("Received NATS backend.stop event", "backend", req.Backend)
			s.stopBackend(req.Backend)
		} else {
			xlog.Info("Received NATS backend.stop event (all)")
			s.stopAllBackends()
		}
		replyJSON(reply, messaging.BackendStopReply{Success: true})
	}()
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./core/services/worker/ -count=1 -args -ginkgo.focus="handleBackendStop"`
Expected: PASS (2 specs)

- [ ] **Step 5: Run the whole worker package**

Run: `go test ./core/services/worker/ -count=1`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add core/services/worker/lifecycle.go core/services/worker/lifecycle_stop_test.go
git commit -m "feat(worker): ack backend.stop via request-reply after process kill"
```

---

### Task 3: Worker — bound the Free() call in stopBackendExact

`stopBackendExact` (`core/services/worker/supervisor.go:227-252`) calls `client.Free(context.Background())` with no deadline before killing the process. A backend wedged mid-LoadModel can hold its gRPC server busy indefinitely, which would stall both the kill and the new ack. Bound it.

No test: the gRPC client is constructed inline (`grpc.NewClientWithToken`) with no injection seam — accepted as untestable-without-refactor in the spec. The existing suite guards against regressions.

**Files:**
- Modify: `core/services/worker/supervisor.go:227-252`

- [ ] **Step 1: Add the timeout**

In `core/services/worker/supervisor.go`, add a constant near the top of the file (below the imports):

```go
// backendFreeTimeout bounds the best-effort gRPC Free() call made before
// killing a backend process. A backend wedged mid-LoadModel can keep its
// gRPC server busy indefinitely; without a bound, the kill — and the
// backend.stop ack the frontend's abandoned-load cleanup blocks on — would
// hang behind the very load being aborted.
const backendFreeTimeout = 10 * time.Second
```

In `stopBackendExact`, replace:

```go
	client := grpc.NewClientWithToken(bp.addr, false, nil, false, s.cfg.RegistrationToken)
	xlog.Debug("Calling Free() before stopping backend", "backend", key)
	if err := client.Free(context.Background()); err != nil {
		xlog.Warn("Free() failed (best-effort)", "backend", key, "error", err)
	}
```

with:

```go
	client := grpc.NewClientWithToken(bp.addr, false, nil, false, s.cfg.RegistrationToken)
	xlog.Debug("Calling Free() before stopping backend", "backend", key)
	freeCtx, cancelFree := context.WithTimeout(context.Background(), backendFreeTimeout)
	if err := client.Free(freeCtx); err != nil {
		xlog.Warn("Free() failed (best-effort)", "backend", key, "error", err)
	}
	cancelFree()
```

- [ ] **Step 2: Build and run the worker suite**

Run: `go build ./core/... ; go test ./core/services/worker/ -count=1`
Expected: build OK, tests PASS

- [ ] **Step 3: Commit**

```bash
git add core/services/worker/supervisor.go
git commit -m "fix(worker): bound pre-kill Free() so a wedged load cannot stall backend.stop"
```

---

### Task 4: Registry — LoadedReplicaStats must return ReplicaIndex

`LoadedReplicaStats` (`core/services/nodes/registry.go:1105-1136`) selects only `node_id` and `in_flight`; `ReplicaIndex` is always zero. Its comment claims the sole consumer reads only those columns, but `buildPreference` (`router.go:685`) already keys `prefixcache.ReplicaKey` on `s.ReplicaIndex` — a latent bug for multi-replica prefix-cache routing. The abandoned-load coalescing guard (Task 7) also needs the real index. Fix the SELECT.

**Files:**
- Modify: `core/services/nodes/registry.go:1105-1136`
- Test: `core/services/nodes/registry_test.go` (existing `Describe("LoadedReplicaStats")` at line ~500)

- [ ] **Step 1: Write the failing test**

Inside the existing `Describe("LoadedReplicaStats")` block in `core/services/nodes/registry_test.go` (after the `"filters to the candidate node set when provided"` It), add:

```go
		It("returns the replica index for each loaded replica", func() {
			// A second replica of the same model on n2, slot 1.
			Expect(registry.SetNodeModel(context.Background(), n2.ID, "stats-model", 1, "loaded", "10.0.0.81:6001", 0, "")).To(Succeed())

			stats, err := registry.LoadedReplicaStats(context.Background(), "stats-model", []string{n2.ID})
			Expect(err).ToNot(HaveOccurred())

			indices := []int{}
			for _, s := range stats {
				indices = append(indices, s.ReplicaIndex)
			}
			Expect(indices).To(ConsistOf(0, 1),
				"LoadedReplicaStats must surface replica_index — prefix-cache keys and abandoned-load cleanup both depend on it")
		})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./core/services/nodes/ -count=1 -args -ginkgo.focus="returns the replica index"`
Expected: FAIL — both indices come back 0 because the SELECT omits `replica_index`.

- [ ] **Step 3: Fix the SELECT**

In `core/services/nodes/registry.go`, `LoadedReplicaStats`, replace:

```go
	// Narrow to only the columns the sole consumer (router buildPreference)
	// reads: NodeID and InFlight. The other ReplicaCandidate fields stay at
	// their zero value, which the consumer does not read. This avoids the
	// JOIN-side available_vram fetch and the extra column transfer.
	var rows []row
	err := q.Select("node_models.node_id AS node_id, node_models.in_flight AS in_flight").
		Scan(&rows).Error
```

with:

```go
	// Narrow to the columns the consumers read: NodeID, ReplicaIndex, and
	// InFlight. buildPreference keys prefixcache.ReplicaKey on ReplicaIndex
	// and the abandoned-load cleanup guard matches on it, so it must be real
	// — selecting only node_id+in_flight silently pinned every replica key
	// to 0. The remaining ReplicaCandidate fields stay at their zero value;
	// this still avoids the JOIN-side available_vram fetch.
	var rows []row
	err := q.Select("node_models.node_id AS node_id, node_models.replica_index AS replica_index, node_models.in_flight AS in_flight").
		Scan(&rows).Error
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./core/services/nodes/ -count=1 -args -ginkgo.focus="LoadedReplicaStats"`
Expected: PASS (all 5 specs in the Describe, including the new one)

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/registry.go core/services/nodes/registry_test.go
git commit -m "fix(nodes): LoadedReplicaStats returns real replica_index"
```

---

### Task 5: Frontend — StopBackendAndWait on NodeCommandSender

Add the acked-stop client method. Same subject and request payload as `StopBackend`, but `RequestJSON` instead of `Publish`, with a 30s ack deadline (covers the worker's 10s-bounded Free + proc.Stop + NATS delivery).

**Files:**
- Modify: `core/services/nodes/unloader.go` (interface at :36-43, adapter)
- Modify: `core/services/nodes/router_test.go:400-482` (`fakeUnloader`)
- Test: `core/services/nodes/unloader_test.go`

- [ ] **Step 1: Write the failing test**

In `core/services/nodes/unloader_test.go`, add a new top-level Describe (the `newScriptedMessagingClient` helper already exists in this package — see its use at line ~269):

```go
var _ = Describe("RemoteUnloaderAdapter StopBackendAndWait", func() {
	It("sends an acked backend.stop request with the exact process key", func() {
		mc := newScriptedMessagingClient()
		mc.scriptReply(messaging.SubjectNodeBackendStop("n1"), messaging.BackendStopReply{Success: true})
		adapter := NewRemoteUnloaderAdapter(nil, mc, time.Second, time.Second)

		Expect(adapter.StopBackendAndWait("n1", "my-model#0")).To(Succeed())

		Expect(mc.calls).To(HaveLen(1))
		Expect(mc.calls[0].Subject).To(Equal(messaging.SubjectNodeBackendStop("n1")))
		Expect(mc.calls[0].Timeout).To(Equal(30 * time.Second))
		var req messaging.BackendStopRequest
		Expect(json.Unmarshal(mc.calls[0].Data, &req)).To(Succeed())
		Expect(req.Backend).To(Equal("my-model#0"))
	})

	It("surfaces a worker-reported failure", func() {
		mc := newScriptedMessagingClient()
		mc.scriptReply(messaging.SubjectNodeBackendStop("n1"), messaging.BackendStopReply{Success: false, Error: "kill failed"})
		adapter := NewRemoteUnloaderAdapter(nil, mc, time.Second, time.Second)

		Expect(adapter.StopBackendAndWait("n1", "m#0")).To(MatchError(ContainSubstring("kill failed")))
	})

	It("returns the timeout error when an old worker never acks", func() {
		mc := newScriptedMessagingClient()
		mc.scriptErr(messaging.SubjectNodeBackendStop("n1"), nats.ErrTimeout)
		adapter := NewRemoteUnloaderAdapter(nil, mc, time.Second, time.Second)

		Expect(adapter.StopBackendAndWait("n1", "m#0")).To(HaveOccurred())
	})
})
```

(`newScriptedMessagingClient` lives in `managers_distributed_test.go:69`; its `calls` field is `[]requestCall` — the struct defined in `unloader_test.go:63` with `Subject`/`Data`/`Timeout` fields — so the assertions above compile as written.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./core/services/nodes/ -count=1 -args -ginkgo.focus="StopBackendAndWait"`
Expected: COMPILE FAILURE — `StopBackendAndWait` is undefined. TDD red.

- [ ] **Step 3: Implement the interface method, adapter, and fake**

In `core/services/nodes/unloader.go`, extend the `NodeCommandSender` interface (after `StopBackend`):

```go
	// StopBackendAndWait stops a backend process on a node and waits for the
	// worker's ack, so the caller can hold a coalescing lock until the
	// process is confirmed dead. backend should be the exact processKey
	// (`modelID#replica`) — a bare model ID prefix-matches every replica on
	// the worker. Old workers execute the stop but never ack; the call then
	// returns a timeout error and the caller should degrade to
	// fire-and-forget semantics (log and continue).
	StopBackendAndWait(nodeID, backend string) error
```

Add to the adapter (below `StopBackend`):

```go
// backendStopAckTimeout bounds the synchronous backend.stop request-reply:
// the worker's bounded Free() (10s) + proc.Stop + NATS delivery.
const backendStopAckTimeout = 30 * time.Second

// StopBackendAndWait tells a worker to stop a backend process and waits for
// the ack. Same subject and payload as StopBackend; only the delivery mode
// differs. Used by the abandoned-load cleanup, which must not release the
// model-load advisory lock while the doomed process is still alive.
func (a *RemoteUnloaderAdapter) StopBackendAndWait(nodeID, backend string) error {
	subject := messaging.SubjectNodeBackendStop(nodeID)
	xlog.Info("Sending NATS backend.stop (acked)", "nodeID", nodeID, "backend", backend)

	reply, err := messaging.RequestJSON[messaging.BackendStopRequest, messaging.BackendStopReply](
		a.nats, subject, messaging.BackendStopRequest{Backend: backend}, backendStopAckTimeout)
	if err != nil {
		return err
	}
	if !reply.Success {
		return fmt.Errorf("backend.stop on node %s: %s", nodeID, reply.Error)
	}
	return nil
}
```

In `core/services/nodes/router_test.go`, extend `fakeUnloader`: add fields next to `stopCalls` (line ~418):

```go
	stopAndWaitCalls []string // "nodeID:processKey"
	stopAndWaitErr   error
```

and the method (next to `StopBackend`, line ~474):

```go
func (f *fakeUnloader) StopBackendAndWait(nodeID, backend string) error {
	f.mu.Lock()
	f.stopAndWaitCalls = append(f.stopAndWaitCalls, nodeID+":"+backend)
	f.mu.Unlock()
	return f.stopAndWaitErr
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./core/services/nodes/ -count=1 -args -ginkgo.focus="StopBackendAndWait"`
Expected: PASS (3 specs)

- [ ] **Step 5: Build everything (interface implementers must compile)**

Run: `go build ./core/...`
Expected: OK. If any other `NodeCommandSender` implementation fails to compile (search `NodeCommandSender` if so), add the same one-line recording/delegating method there.

- [ ] **Step 6: Commit**

```bash
git add core/services/nodes/unloader.go core/services/nodes/unloader_test.go core/services/nodes/router_test.go
git commit -m "feat(nodes): StopBackendAndWait — acked backend.stop for load cleanup"
```

---

### Task 6: Frontend — placement struct refactor (behavior-preserving)

`scheduleNewModel` (`core/services/nodes/router.go:887`) returns five values and keeps `reservedGPUs`/`perGPUReserve` as locals, so post-install failure paths can't roll reservations back. Refactor to a `placement` struct. No behavior change; existing tests stay green.

**Files:**
- Modify: `core/services/nodes/router.go`

- [ ] **Step 1: Define the struct and change the signature**

Above `scheduleNewModel` in `core/services/nodes/router.go`, add:

```go
// placement is scheduleNewModel's result: where a new model load goes and
// what bookkeeping it holds. reservedGPUs/perGPUReserve record the per-GPU
// soft reservations taken at scheduling time so post-install failure paths
// (abandoned loads) can roll them back explicitly instead of waiting for
// the next worker heartbeat to reset them.
type placement struct {
	node         *BackendNode
	addr         string
	replicaIndex int
	// gpuSet is the set of physical GPU indices the model was pinned to
	// (empty for legacy/CPU placement).
	gpuSet []int
	// reservedGPUs are the GPU indices where ReserveVRAMOnGPU succeeded;
	// perGPUReserve is the bytes reserved on each.
	reservedGPUs  []int
	perGPUReserve uint64
}
```

Change the signature from:

```go
func (r *SmartRouter) scheduleNewModel(ctx context.Context, backendType, modelID string, modelOpts *pb.ModelOptions) (*BackendNode, string, int, []int, error) {
```

to:

```go
func (r *SmartRouter) scheduleNewModel(ctx context.Context, backendType, modelID string, modelOpts *pb.ModelOptions) (*placement, error) {
```

Mechanically update every `return` in the function body:
- every error return `return nil, "", 0, nil, <err-expr>` becomes `return nil, <err-expr>`
- the success return at the end (`return node, addr, replicaIdx, gpuSet, nil`) becomes:

```go
	return &placement{
		node:          node,
		addr:          addr,
		replicaIndex:  replicaIdx,
		gpuSet:        gpuSet,
		reservedGPUs:  reservedGPUs,
		perGPUReserve: perGPUReserve,
	}, nil
```

(`reservedGPUs` and `perGPUReserve` are the existing locals declared at router.go:1175-1176 — they are now carried out instead of dropped.)

- [ ] **Step 2: Update the call site in scheduleAndLoad**

In `scheduleAndLoad` (router.go:294-300), replace:

```go
	node, backendAddr, replicaIndex, gpuSet, err := r.scheduleNewModel(ctx, backendType, trackingKey, modelOpts)
	if err != nil {
		return nil, fmt.Errorf("no available nodes: %w", err)
	}
```

with:

```go
	p, err := r.scheduleNewModel(ctx, backendType, trackingKey, modelOpts)
	if err != nil {
		return nil, fmt.Errorf("no available nodes: %w", err)
	}
	node, backendAddr, replicaIndex, gpuSet := p.node, p.addr, p.replicaIndex, p.gpuSet
```

The rest of `scheduleAndLoad`'s body keeps its existing local names untouched.

- [ ] **Step 3: Build and run the full nodes package**

Run: `go build ./core/... ; go test ./core/services/nodes/ -count=1`
Expected: build OK; tests PASS except the 11 known `staging_keys_test.go` Windows failures.

- [ ] **Step 4: Commit**

```bash
git add core/services/nodes/router.go
git commit -m "refactor(nodes): scheduleNewModel returns a placement struct carrying reservations"
```

---

### Task 7: Frontend — cleanupAbandonedLoad + arming in scheduleAndLoad (the core)

**Files:**
- Create: `core/services/nodes/abandoned_load_cleanup_test.go`
- Modify: `core/services/nodes/router.go` (`scheduleAndLoad`, new helper)
- Modify: `core/services/nodes/router_test.go` (`fakeModelRouter`: record GPU reserve/release, configurable replica index)

- [ ] **Step 1: Add fake recording seams**

In `core/services/nodes/router_test.go`:

In the `fakeModelRouter` struct field list (near `loadedReplicaStatsByName`, line ~120), add:

```go
	// nextFreeReplica is returned by NextFreeReplicaIndex (default 0).
	nextFreeReplica int

	// reserveGPUCalls / releaseGPUCalls record per-GPU soft reservation
	// traffic as "nodeID:gpuIndex:bytes" for cleanup assertions.
	reserveGPUCalls []string
	releaseGPUCalls []string
```

Replace the three corresponding methods (lines ~220 and ~343-349):

```go
func (f *fakeModelRouter) NextFreeReplicaIndex(_ context.Context, _, _ string, _ int) (int, error) {
	return f.nextFreeReplica, nil
}

func (f *fakeModelRouter) ReserveVRAMOnGPU(_ context.Context, nodeID string, gpuIndex int, bytes uint64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.reserveGPUCalls = append(f.reserveGPUCalls, fmt.Sprintf("%s:%d:%d", nodeID, gpuIndex, bytes))
	return nil
}

func (f *fakeModelRouter) ReleaseVRAMOnGPU(_ context.Context, nodeID string, gpuIndex int, bytes uint64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.releaseGPUCalls = append(f.releaseGPUCalls, fmt.Sprintf("%s:%d:%d", nodeID, gpuIndex, bytes))
	return nil
}
```

- [ ] **Step 2: Write the failing tests**

Create `core/services/nodes/abandoned_load_cleanup_test.go`:

```go
package nodes

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/nats-io/nats.go"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/mudler/LocalAI/core/services/messaging"
	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
)

// failingStager is a FileStager whose uploads always fail — used to force
// the staging-error path of scheduleAndLoad after the worker process was
// already spawned by installBackendOnNode.
type failingStager struct{}

func (f *failingStager) EnsureRemote(_ context.Context, _, _, _ string) (string, error) {
	return "", errors.New("simulated staging failure")
}
func (f *failingStager) FetchRemote(_ context.Context, _, _, _ string) error      { return nil }
func (f *failingStager) FetchRemoteByKey(_ context.Context, _, _, _ string) error { return nil }
func (f *failingStager) AllocRemoteTemp(_ context.Context, _ string) (string, error) {
	return "", errors.New("not implemented")
}
func (f *failingStager) StageRemoteToStore(_ context.Context, _, _, _ string) error { return nil }
func (f *failingStager) ListRemoteDir(_ context.Context, _, _ string) ([]string, error) {
	return nil, nil
}

var _ = Describe("abandoned-load cleanup (scheduleAndLoad)", func() {
	const estimate = uint64(8 << 30) // 8 GiB — fits the single 24 GiB GPU below

	var (
		node     *BackendNode
		reg      *fakeModelRouter
		backend  *stubBackend
		factory  *stubClientFactory
		unloader *fakeUnloader
	)

	BeforeEach(func() {
		node = &BackendNode{ID: "n1", Name: "worker-1", Address: "10.0.0.1:50051"}
		reg = &fakeModelRouter{
			findAndLockErr: errors.New("not found"), // force the cold-load path
			candidateNodes: []BackendNode{*node},
			nodeGPUs:       []NodeGPU{{NodeID: "n1", GPUIndex: 0, TotalVRAM: 24 << 30, FreeVRAM: 24 << 30}},
			findIdleNode:   node,
		}
		backend = &stubBackend{}
		factory = &stubClientFactory{client: backend}
		unloader = &fakeUnloader{
			installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:9001"},
		}
	})

	newRouter := func(stager FileStager) *SmartRouter {
		return NewSmartRouter(reg, SmartRouterOptions{
			Unloader:      unloader,
			ClientFactory: factory,
			FileStager:    stager,
			VRAMEstimator: func(context.Context, string, *pb.ModelOptions) uint64 { return estimate },
		})
	}

	route := func(r *SmartRouter) error {
		_, err := r.Route(context.Background(), "m", "models/m.safetensors", "diffusers", &pb.ModelOptions{}, false)
		return err
	}

	expectedRelease := fmt.Sprintf("n1:0:%d", estimate)

	It("kills the exact worker process and rolls back the reservation when LoadModel errors", func() {
		backend.loadErr = errors.New("context deadline exceeded")

		err := route(newRouter(nil))
		Expect(err).To(MatchError(ContainSubstring("loading model")))

		// Exact processKey — never the bare model ID, which would
		// prefix-match and kill sibling replicas on the worker.
		Expect(unloader.stopAndWaitCalls).To(ConsistOf("n1:m#0"))
		Expect(unloader.stopCalls).To(BeEmpty(), "cleanup must use the acked stop")
		Expect(reg.releaseGPUCalls).To(ConsistOf(expectedRelease))
	})

	It("kills the process when the backend reports a failed load", func() {
		backend.loadResult = &pb.Result{Success: false, Message: "CUDA out of memory"}

		err := route(newRouter(nil))
		Expect(err).To(MatchError(ContainSubstring("CUDA out of memory")))
		Expect(unloader.stopAndWaitCalls).To(ConsistOf("n1:m#0"))
		Expect(reg.releaseGPUCalls).To(ConsistOf(expectedRelease))
	})

	It("targets the allocated replica slot, not slot zero", func() {
		reg.nextFreeReplica = 1
		backend.loadErr = errors.New("boom")

		Expect(route(newRouter(nil))).To(HaveOccurred())
		Expect(unloader.stopAndWaitCalls).To(ConsistOf("n1:m#1"),
			"abandoning m#1 must never touch m#0 or any other sibling replica")
	})

	It("cleans up when staging fails after the process was spawned", func() {
		// ModelFile must exist on disk: stageModelFiles silently clears
		// non-existent paths instead of failing.
		tmp, err := os.CreateTemp("", "model-*.safetensors")
		Expect(err).ToNot(HaveOccurred())
		DeferCleanup(func() { os.Remove(tmp.Name()) })
		_, err = tmp.WriteString("weights")
		Expect(err).ToNot(HaveOccurred())
		Expect(tmp.Close()).To(Succeed())

		r := newRouter(&failingStager{})
		_, err = r.Route(context.Background(), "m", "m.safetensors", "diffusers",
			&pb.ModelOptions{Model: "m.safetensors", ModelFile: tmp.Name()}, false)
		Expect(err).To(MatchError(ContainSubstring("staging")))

		Expect(unloader.stopAndWaitCalls).To(ConsistOf("n1:m#0"))
		Expect(reg.releaseGPUCalls).To(ConsistOf(expectedRelease))
	})

	It("does not clean up after a successful load", func() {
		backend.loadResult = &pb.Result{Success: true}

		Expect(route(newRouter(nil))).To(Succeed())
		Expect(reg.setCalls).To(HaveLen(1), "sanity: the load was recorded")
		Expect(unloader.stopAndWaitCalls).To(BeEmpty())
		Expect(reg.releaseGPUCalls).To(BeEmpty())
	})

	It("skips the kill when a concurrent load registered the same replica, but still rolls back the reservation", func() {
		// The reconciler's ScheduleAndLoadModel runs without the advisory
		// lock; if its racing load of the same (node, model, replica)
		// succeeded, the process is a live serving replica — killing it
		// would take down real traffic.
		reg.loadedReplicaStatsByName = map[string][]ReplicaCandidate{
			"m": {{NodeID: "n1", ReplicaIndex: 0}},
		}
		backend.loadErr = errors.New("context deadline exceeded")

		Expect(route(newRouter(nil))).To(HaveOccurred())
		Expect(unloader.stopAndWaitCalls).To(BeEmpty())
		Expect(reg.releaseGPUCalls).To(ConsistOf(expectedRelease))
	})

	It("still returns the original load error when the stop ack times out (old worker)", func() {
		unloader.stopAndWaitErr = nats.ErrTimeout
		backend.loadErr = errors.New("context deadline exceeded")

		err := route(newRouter(nil))
		Expect(err).To(MatchError(ContainSubstring("loading model")),
			"cleanup failures must never mask the load error")
		Expect(unloader.stopAndWaitCalls).To(HaveLen(1))
	})
})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `go test ./core/services/nodes/ -count=1 -args -ginkgo.focus="abandoned-load cleanup"`
Expected: FAIL — `stopAndWaitCalls` stays empty and `releaseGPUCalls` stays empty on every failure-path spec (no cleanup exists yet). The success-path and old-worker specs may pass trivially; the other five must be red.

- [ ] **Step 4: Implement cleanupAbandonedLoad and arm it**

In `core/services/nodes/router.go`, add below `bumpEstimateOnLoadFailure`:

```go
// abandonedCleanupTimeout bounds the registry calls made while cleaning up
// an abandoned load (the acked stop carries its own 30s NATS deadline).
const abandonedCleanupTimeout = 45 * time.Second

// cleanupAbandonedLoad kills the worker-side backend process for a load that
// was abandoned after the process was spawned — staging failure, LoadModel
// timeout / caller disconnect / transport error, or backend-reported failure
// — and rolls back the per-GPU soft reservations. Without this the worker
// keeps loading: it completes invisibly, holds RAM/VRAM with no NodeModel
// row, and both the stale-replica reaper and per-GPU accounting are blind to
// it (2026-06-03 incident: OOMKilled worker + 7.1GB orphan on GPU0).
//
// Runs on a fresh context: the request ctx is typically already cancelled
// (timeout / disconnect) — that is the point. The stop is synchronous
// (acked): Route holds the model-load advisory lock through this call, so a
// queued request for the same model cannot re-install until the process is
// confirmed dead. Best-effort throughout: failures are logged, never
// propagated — the load error already on its way to the caller stays the
// authoritative outcome.
func (r *SmartRouter) cleanupAbandonedLoad(p *placement, trackingKey string) {
	ctx, cancel := context.WithTimeout(context.Background(), abandonedCleanupTimeout)
	defer cancel()

	processKey := fmt.Sprintf("%s#%d", trackingKey, p.replicaIndex)

	// Coalescing guard: the reconciler's ScheduleAndLoadModel runs without
	// the advisory lock, so a racing load of the same (node, model, replica)
	// may have already succeeded and registered. Killing now would take down
	// a live replica — skip the kill, keep the reservation rollback.
	kill := true
	if stats, err := r.registry.LoadedReplicaStats(ctx, trackingKey, []string{p.node.ID}); err == nil {
		for _, s := range stats {
			if s.NodeID == p.node.ID && s.ReplicaIndex == p.replicaIndex {
				kill = false
				break
			}
		}
	}

	switch {
	case !kill:
		xlog.Info("Abandoned-load cleanup: replica registered as loaded by a concurrent request, skipping kill",
			"node", p.node.Name, "processKey", processKey)
	case r.unloader == nil:
		// Non-distributed configuration — nothing to send the stop through.
	default:
		xlog.Info("Abandoned-load cleanup: stopping worker-side backend process",
			"node", p.node.Name, "processKey", processKey)
		if err := r.unloader.StopBackendAndWait(p.node.ID, processKey); err != nil {
			// Old workers execute the stop but never ack (timeout); real
			// failures surface on the next placement attempt anyway.
			xlog.Warn("Abandoned-load cleanup: stop not acked",
				"node", p.node.Name, "processKey", processKey, "error", err)
		}
	}

	// Roll back the per-GPU soft reservations. Underflow-guarded in the
	// registry and reset by the worker's next heartbeat regardless, so this
	// only matters for failures inside the first heartbeat window — but it
	// keeps the columns accurate, symmetric with the install-failure rollback.
	for _, gi := range p.reservedGPUs {
		if err := r.registry.ReleaseVRAMOnGPU(ctx, p.node.ID, gi, p.perGPUReserve); err != nil {
			xlog.Debug("Abandoned-load cleanup: reservation release failed",
				"node", p.node.ID, "gpu", gi, "bytes", p.perGPUReserve, "error", err)
		}
	}
}
```

In `scheduleAndLoad`, directly after the placement destructuring added in Task 6 (`node, backendAddr, replicaIndex, gpuSet := ...`), arm the cleanup:

```go
	// The worker-side process exists from here on (installBackendOnNode ran
	// inside scheduleNewModel). If the load is abandoned on any failure path
	// below — staging error, LoadModel timeout/disconnect/error, or
	// backend-reported failure — kill that process and roll back the
	// reservations; otherwise it keeps loading invisibly with no registry
	// row (2026-06-03 incident).
	committed := false
	defer func() {
		if !committed {
			r.cleanupAbandonedLoad(p, trackingKey)
		}
	}()
```

Then, directly after the `if loadOpts != nil { ... }` LoadModel block (immediately before the `// Record the model as loaded on this node ...` comment at router.go:366), disarm it:

```go
	// Load committed: every path past here leaves a live, registered (or at
	// worst registry-lagging) replica that the normal unload paths own.
	committed = true
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `go test ./core/services/nodes/ -count=1 -args -ginkgo.focus="abandoned-load cleanup"`
Expected: PASS (7 specs)

- [ ] **Step 6: Run the full nodes package**

Run: `go test ./core/services/nodes/ -count=1`
Expected: PASS except the 11 known `staging_keys_test.go` Windows failures.

- [ ] **Step 7: Commit**

```bash
git add core/services/nodes/router.go core/services/nodes/router_test.go core/services/nodes/abandoned_load_cleanup_test.go
git commit -m "fix(nodes): kill abandoned remote backend loads and roll back GPU reservations"
```

---

### Task 8: Full verification

- [ ] **Step 1: Build everything**

Run: `go build ./...`
Expected: OK

- [ ] **Step 2: Vet the touched packages**

Run: `go vet ./core/services/nodes/... ./core/services/worker/... ./core/services/messaging/...`
Expected: clean

- [ ] **Step 3: Run the touched packages' suites**

Run: `go test ./core/services/nodes/ ./core/services/worker/ ./core/services/messaging/ -count=1`
Expected: PASS except the 11 known `staging_keys_test.go` Windows failures. Record the exact failure list and confirm it matches the known set before claiming green.

- [ ] **Step 4: Push and open the PR**

Use the superpowers:finishing-a-development-branch skill. PR target: `custom`. Reference the spec and the 2026-06-03 incident in the description. After merge, deployment is manual: `gh workflow run "fork: build + push to GHCR" --ref custom -f tag=custom` (~10m), then Keel auto-rolls the cluster.
