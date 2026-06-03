# Per-GPU VRAM-Aware Scheduling & Multi-GPU Model Placement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Schedule and place model backends per-GPU instead of per-node-aggregate, so independent backends spread across a node's GPUs and large models can span multiple GPUs on one node — driven by a VRAM estimate that works for diffusers/SDXL models.

**Architecture:** Per-GPU free VRAM is reported in the worker heartbeat and stored in a new `node_gpus` table. The scheduler computes a VRAM estimate (measured → GGUF/HF metadata → file-size heuristic), picks a node and a specific GPU set (best-fit on one GPU, else fewest-GPUs spanning), soft-reserves per-GPU, and tells the worker which physical GPU indices to pin via `CUDA_VISIBLE_DEVICES`. The multi-GPU knob (`TensorParallelSize`/`TensorSplit`) in `ModelOptions` is set to the GPU-set size. After a successful load the frontend measures the real per-GPU delta and caches it to correct the heuristic.

**Tech Stack:** Go, GORM (Postgres), NATS (request/reply), gRPC (`pb.ModelOptions`), Ginkgo/Gomega tests, testcontainers Postgres (`testutil.SetupTestDB`).

**Spec:** `docs/superpowers/specs/2026-06-03-per-gpu-vram-scheduling-design.md`

**Conventions (per repo):** tabs for indentation in `.go`? No — Go uses gofmt (tabs already). Run `gofmt -w` on touched files. Tests are Ginkgo/Gomega; run with `go test ./<pkg>/...`. Commit after every green task.

---

## File Structure

**New files:**
- `pkg/vram/heuristic.go` — `EstimateHeuristic(modelPath string) uint64` (file-size seed for unparseable models). Responsibility: size a model from on-disk weights when metadata estimation can't.
- `pkg/vram/heuristic_test.go` — unit tests for the heuristic.
- `core/services/nodes/gpuplan.go` — `planGPUSet(gpus []NodeGPU, estimate uint64) ([]int, bool)` pure placement function (best-fit / fewest-GPU spanning). Responsibility: decide the GPU index set on one node, no DB access.
- `core/services/nodes/gpuplan_test.go` — table-driven unit tests for `planGPUSet`.

**Modified files:**
- `pkg/xsysinfo/gpu.go` — (no change; `GetGPUMemoryUsage()` already returns `[]GPUMemoryInfo`).
- `core/services/nodes/registry.go` — new models `NodeGPU`, `ModelVRAMEstimate`; migration; per-GPU reserve/release; `NodeGPUs`, `GetModelVRAMEstimate`, `UpsertModelVRAMEstimate` accessors; `HeartbeatUpdate.GPUs`; `Heartbeat` upserts `node_gpus`; `NodeModel.GPUIndices` column.
- `core/http/endpoints/localai/nodes.go` — heartbeat ingest accepts `GPUs`.
- `core/services/worker/registration.go` — `heartbeatBody()` emits `gpus[]`.
- `core/services/messaging/subjects.go` — `BackendInstallRequest.GPUIndices`.
- `core/services/nodes/unloader.go` — `NodeCommandSender.InstallBackend` signature gains `gpuIndices []int`; impl sets it on the request.
- `core/services/worker/install.go` + `core/services/worker/supervisor.go` + `pkg/model/process.go` — thread `GPUIndices` to process start; set `CUDA_VISIBLE_DEVICES`.
- `core/services/nodes/router.go` — `estimateModelVRAM` lookup order; `scheduleNewModel` per-GPU placement, per-GPU reservation, `applyGPUSet` to `ModelOptions`, bounded retries + terminal error; `scheduleAndLoad` measurement.

---

# Phase 1 — Per-GPU data foundation

Goal of phase: workers report per-GPU free VRAM; registry stores it in `node_gpus` with per-GPU soft reservations. Additive and inert (nothing reads it for placement yet). Phase is green when all `core/services/...` and `pkg/...` tests pass.

### Task 1.1: `NodeGPU` model + migration

**Files:**
- Modify: `core/services/nodes/registry.go` (struct near `NodeModel` ~line 96; migration call ~line 265)
- Test: `core/services/nodes/registry_test.go`

- [ ] **Step 1: Write the failing test**

Add inside the `Describe("NodeRegistry", ...)` block in `registry_test.go`:

```go
	Describe("NodeGPU migration", func() {
		It("creates the node_gpus table and round-trips a row", func() {
			gpu := NodeGPU{NodeID: "node-1", GPUIndex: 0, TotalVRAM: 12_000_000_000, FreeVRAM: 11_000_000_000}
			Expect(db.Create(&gpu).Error).To(Succeed())

			var got NodeGPU
			Expect(db.Where("node_id = ? AND gpu_index = ?", "node-1", 0).First(&got).Error).To(Succeed())
			Expect(got.TotalVRAM).To(Equal(uint64(12_000_000_000)))
			Expect(got.FreeVRAM).To(Equal(uint64(11_000_000_000)))
			Expect(got.ReservedVRAM).To(Equal(uint64(0)))
		})
	})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v` (Ginkgo runs under the package's `Test*` entrypoint)
Expected: FAIL — `undefined: NodeGPU`.

- [ ] **Step 3: Add the model and migration**

In `registry.go`, after the `NodeModel` struct, add:

```go
// NodeGPU is one physical GPU on a backend node. Free/total VRAM are refreshed
// by the worker heartbeat (the worker is the source of truth). ReservedVRAM is
// the per-GPU soft, in-tick reservation the scheduler deducts when it assigns a
// model to this GPU; the worker's next heartbeat clears it back to 0.
type NodeGPU struct {
	NodeID       string    `gorm:"primaryKey;size:36" json:"node_id"`
	GPUIndex     int       `gorm:"primaryKey;column:gpu_index" json:"gpu_index"`
	TotalVRAM    uint64    `gorm:"column:total_vram" json:"total_vram"`
	FreeVRAM     uint64    `gorm:"column:free_vram" json:"free_vram"`
	ReservedVRAM uint64    `gorm:"column:reserved_vram;default:0" json:"reserved_vram"`
	UpdatedAt    time.Time `json:"updated_at"`
}

func (NodeGPU) TableName() string { return "node_gpus" }
```

In `NewNodeRegistry`, add `&NodeGPU{}` to the `AutoMigrate` list (line ~265):

```go
		return db.AutoMigrate(&BackendNode{}, &NodeModel{}, &NodeLabel{}, &ModelSchedulingConfig{}, &PendingBackendOp{}, &ModelLoadInfo{}, &NodeGPU{})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/registry.go core/services/nodes/registry_test.go
git commit -m "feat(nodes): add NodeGPU model and migration for per-GPU VRAM"
```

---

### Task 1.2: Per-GPU reserve/release + `NodeGPUs` accessor

**Files:**
- Modify: `core/services/nodes/registry.go` (near `ReserveVRAM`/`ReleaseVRAM` ~line 519)
- Test: `core/services/nodes/registry_test.go`

- [ ] **Step 1: Write the failing test**

```go
	Describe("per-GPU reservation", func() {
		BeforeEach(func() {
			Expect(db.Create(&NodeGPU{NodeID: "n", GPUIndex: 0, TotalVRAM: 12e9, FreeVRAM: 12e9}).Error).To(Succeed())
		})
		It("ReserveVRAMOnGPU deducts from effectively-free and rejects when insufficient", func() {
			Expect(registry.ReserveVRAMOnGPU(context.Background(), "n", 0, 8e9)).To(Succeed())
			err := registry.ReserveVRAMOnGPU(context.Background(), "n", 0, 8e9)
			Expect(err).To(MatchError(ErrInsufficientVRAM))
		})
		It("ReleaseVRAMOnGPU gives the reservation back", func() {
			Expect(registry.ReserveVRAMOnGPU(context.Background(), "n", 0, 8e9)).To(Succeed())
			Expect(registry.ReleaseVRAMOnGPU(context.Background(), "n", 0, 8e9)).To(Succeed())
			Expect(registry.ReserveVRAMOnGPU(context.Background(), "n", 0, 8e9)).To(Succeed())
		})
		It("NodeGPUs returns the node's GPUs ordered by index", func() {
			Expect(db.Create(&NodeGPU{NodeID: "n", GPUIndex: 1, TotalVRAM: 12e9, FreeVRAM: 4e9}).Error).To(Succeed())
			gpus, err := registry.NodeGPUs(context.Background(), "n")
			Expect(err).ToNot(HaveOccurred())
			Expect(gpus).To(HaveLen(2))
			Expect(gpus[0].GPUIndex).To(Equal(0))
			Expect(gpus[1].GPUIndex).To(Equal(1))
		})
	})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: FAIL — `undefined: registry.ReserveVRAMOnGPU`.

- [ ] **Step 3: Implement the three methods**

In `registry.go`, after `ReleaseVRAM`:

```go
// ReserveVRAMOnGPU atomically deducts a soft reservation from one GPU's
// effectively-free VRAM (free - reserved). Mirrors ReserveVRAM but per-GPU.
func (r *NodeRegistry) ReserveVRAMOnGPU(ctx context.Context, nodeID string, gpuIndex int, bytes uint64) error {
	if bytes == 0 {
		return nil
	}
	res := r.db.WithContext(ctx).Model(&NodeGPU{}).
		Where("node_id = ? AND gpu_index = ? AND (free_vram - reserved_vram) >= ?", nodeID, gpuIndex, bytes).
		UpdateColumn("reserved_vram", gorm.Expr("reserved_vram + ?", bytes))
	if res.Error != nil {
		return fmt.Errorf("reserving %d bytes on node %s gpu %d: %w", bytes, nodeID, gpuIndex, res.Error)
	}
	if res.RowsAffected == 0 {
		return ErrInsufficientVRAM
	}
	return nil
}

// ReleaseVRAMOnGPU returns a per-GPU soft reservation (rollback on failed load).
func (r *NodeRegistry) ReleaseVRAMOnGPU(ctx context.Context, nodeID string, gpuIndex int, bytes uint64) error {
	if bytes == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Model(&NodeGPU{}).
		Where("node_id = ? AND gpu_index = ? AND reserved_vram >= ?", nodeID, gpuIndex, bytes).
		UpdateColumn("reserved_vram", gorm.Expr("reserved_vram - ?", bytes)).Error
}

// NodeGPUs returns the node's GPUs ordered by index.
func (r *NodeRegistry) NodeGPUs(ctx context.Context, nodeID string) ([]NodeGPU, error) {
	var gpus []NodeGPU
	err := r.db.WithContext(ctx).Where("node_id = ?", nodeID).Order("gpu_index ASC").Find(&gpus).Error
	return gpus, err
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/registry.go core/services/nodes/registry_test.go
git commit -m "feat(nodes): per-GPU VRAM reserve/release and NodeGPUs accessor"
```

---

### Task 1.3: Heartbeat carries per-GPU VRAM; registry upserts `node_gpus`

**Files:**
- Modify: `core/services/nodes/registry.go` (`HeartbeatUpdate` ~line 590; `Heartbeat` ~line 600)
- Modify: `core/http/endpoints/localai/nodes.go` (`HeartbeatEndpoint` ~line 339)
- Test: `core/services/nodes/registry_test.go`

- [ ] **Step 1: Write the failing test**

```go
	Describe("heartbeat per-GPU upsert", func() {
		It("upserts node_gpus rows and clears their reservation", func() {
			node := makeNode("hb-node", "10.0.0.9:50051", 24e9)
			Expect(registry.Register(context.Background(), node, true)).To(Succeed())
			Expect(db.Create(&NodeGPU{NodeID: node.ID, GPUIndex: 0, TotalVRAM: 12e9, FreeVRAM: 1e9, ReservedVRAM: 5e9}).Error).To(Succeed())

			gpus := []GPUHeartbeat{
				{Index: 0, TotalVRAM: 12e9, FreeVRAM: 9e9, UsedVRAM: 3e9},
				{Index: 1, TotalVRAM: 12e9, FreeVRAM: 12e9, UsedVRAM: 0},
			}
			Expect(registry.Heartbeat(context.Background(), node.ID, &HeartbeatUpdate{GPUs: gpus})).To(Succeed())

			got, err := registry.NodeGPUs(context.Background(), node.ID)
			Expect(err).ToNot(HaveOccurred())
			Expect(got).To(HaveLen(2))
			Expect(got[0].FreeVRAM).To(Equal(uint64(9e9)))
			Expect(got[0].ReservedVRAM).To(Equal(uint64(0)), "fresh heartbeat clears the soft reservation")
			Expect(got[1].FreeVRAM).To(Equal(uint64(12e9)))
		})
	})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: FAIL — `undefined: GPUHeartbeat` / `HeartbeatUpdate has no field GPUs`.

- [ ] **Step 3: Add the wire type, field, and upsert**

In `registry.go`, add the wire type and extend `HeartbeatUpdate`:

```go
// GPUHeartbeat is one GPU's VRAM reading in a heartbeat payload.
type GPUHeartbeat struct {
	Index     int    `json:"index"`
	TotalVRAM uint64 `json:"total_vram"`
	FreeVRAM  uint64 `json:"free_vram"`
	UsedVRAM  uint64 `json:"used_vram"`
}

type HeartbeatUpdate struct {
	AvailableVRAM *uint64        `json:"available_vram,omitempty"`
	TotalVRAM     *uint64        `json:"total_vram,omitempty"`
	AvailableRAM  *uint64        `json:"available_ram,omitempty"`
	GPUVendor     string         `json:"gpu_vendor,omitempty"`
	GPUs          []GPUHeartbeat `json:"gpus,omitempty"`
}
```

In `Heartbeat`, after the existing `updates` block runs (after the node-row update succeeds, before `return nil`), upsert per-GPU rows:

```go
	if update != nil && len(update.GPUs) > 0 {
		for _, g := range update.GPUs {
			// Worker is source of truth: refresh free/total and clear the
			// in-tick reservation. ON CONFLICT updates the existing row.
			row := NodeGPU{NodeID: nodeID, GPUIndex: g.Index, TotalVRAM: g.TotalVRAM, FreeVRAM: g.FreeVRAM, ReservedVRAM: 0, UpdatedAt: time.Now()}
			if err := db.Clauses(clause.OnConflict{
				Columns:   []clause.Column{{Name: "node_id"}, {Name: "gpu_index"}},
				DoUpdates: clause.AssignmentColumns([]string{"total_vram", "free_vram", "reserved_vram", "updated_at"}),
			}).Create(&row).Error; err != nil {
				xlog.Warn("Failed to upsert node GPU", "node", nodeID, "gpu", g.Index, "error", err)
			}
		}
	}
```

Ensure `registry.go` imports `"gorm.io/gorm/clause"` (add to the import block if absent).

In `core/http/endpoints/localai/nodes.go`, extend the body-present check so a GPU-only heartbeat isn't dropped (~line 339):

```go
		var updatePtr *nodes.HeartbeatUpdate
		if update.AvailableVRAM != nil || update.TotalVRAM != nil || update.AvailableRAM != nil || update.GPUVendor != "" || len(update.GPUs) > 0 {
			updatePtr = &update
		}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/registry.go core/http/endpoints/localai/nodes.go core/services/nodes/registry_test.go
git commit -m "feat(nodes): ingest per-GPU VRAM in heartbeat and upsert node_gpus"
```

---

### Task 1.4: Worker emits `gpus[]` in heartbeat

**Files:**
- Modify: `core/services/worker/registration.go` (`heartbeatBody` ~line 127)
- Test: `core/services/worker/registration_test.go` (create if absent)

- [ ] **Step 1: Write the failing test**

Create/append `core/services/worker/registration_test.go`:

```go
package worker

import (
	"testing"
)

// heartbeatBody must emit a gpus slice whose entries carry per-GPU figures so
// the master can populate node_gpus. We can't fake nvidia-smi here, so assert
// the shape contract: when aggregate VRAM is present, the body has both
// available_vram (compat) and a gpus key.
func TestHeartbeatBodyShape(t *testing.T) {
	cfg := &Config{}
	body := cfg.heartbeatBody()
	// On a CPU-only CI host aggregate VRAM is 0 → gpus omitted, available_ram set.
	if _, hasVRAM := body["available_vram"]; hasVRAM {
		if _, hasGPUs := body["gpus"]; !hasGPUs {
			t.Fatalf("available_vram present but gpus missing: %#v", body)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/worker/ -run TestHeartbeatBodyShape -v`
Expected: FAIL on a GPU host (gpus missing); on CPU CI it passes vacuously — that's acceptable, the real assertion is the implementation below.

- [ ] **Step 3: Emit per-GPU figures**

Replace `heartbeatBody` in `registration.go` with:

```go
func (cfg *Config) heartbeatBody() map[string]any {
	body := map[string]any{}
	gpus := xsysinfo.GetGPUMemoryUsage()
	aggregate := xsysinfo.GetGPUAggregateInfo()
	if aggregate.TotalVRAM > 0 {
		body["available_vram"] = aggregate.FreeVRAM
		// Per-GPU detail lets the scheduler reason per-card instead of summing.
		perGPU := make([]map[string]any, 0, len(gpus))
		for _, g := range gpus {
			perGPU = append(perGPU, map[string]any{
				"index":      g.Index,
				"total_vram": g.TotalVRAM,
				"free_vram":  g.FreeVRAM,
				"used_vram":  g.UsedVRAM,
			})
		}
		body["gpus"] = perGPU
	}

	// CPU-only workers (or workers that lost GPU visibility momentarily):
	// report system RAM so the scheduler still has capacity info.
	if aggregate.TotalVRAM == 0 {
		if ramInfo, err := xsysinfo.GetSystemRAMInfo(); err == nil {
			body["available_ram"] = ramInfo.Available
		}
	}
	return body
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./core/services/worker/ -run TestHeartbeatBodyShape -v`
Expected: PASS. Also build: `go build ./...` → success.

- [ ] **Step 5: Commit**

```bash
git add core/services/worker/registration.go core/services/worker/registration_test.go
git commit -m "feat(worker): emit per-GPU VRAM in heartbeat body"
```

**Phase 1 gate:** `go build ./...` and `go test ./core/services/nodes/... ./core/services/worker/...` green.

---

# Phase 2 — VRAM estimation (heuristic seed + measured cache)

Goal: `estimateModelVRAM` returns a usable non-zero value for diffusers/SDXL models via lookup order `measured → GGUF/HF metadata → heuristic`. Green when `pkg/vram` and `core/services/nodes` tests pass.

### Task 2.1: `vram.EstimateHeuristic`

**Files:**
- Create: `pkg/vram/heuristic.go`
- Test: `pkg/vram/heuristic_test.go`

- [ ] **Step 1: Write the failing test**

Create `pkg/vram/heuristic_test.go`:

```go
package vram

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEstimateHeuristic(t *testing.T) {
	dir := t.TempDir()
	// 4 GiB of "weights" across two safetensors files.
	writeFile(t, filepath.Join(dir, "unet.safetensors"), 3<<30)
	writeFile(t, filepath.Join(dir, "vae.safetensors"), 1<<30)
	// A non-weight file that must be ignored.
	writeFile(t, filepath.Join(dir, "config.json"), 1024)

	got := EstimateHeuristic(dir)
	// 4 GiB * 1.2 + 2 GiB overhead = ~6.8 GiB. Allow a tolerance band.
	min, max := uint64(6<<30), uint64(8<<30)
	if got < min || got > max {
		t.Fatalf("EstimateHeuristic = %d, want within [%d,%d]", got, min, max)
	}
}

func TestEstimateHeuristicSingleFile(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "model.safetensors")
	writeFile(t, f, 2<<30)
	if got := EstimateHeuristic(f); got == 0 {
		t.Fatalf("EstimateHeuristic(file) = 0, want > 0")
	}
}

func writeFile(t *testing.T, path string, size int64) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := f.Truncate(size); err != nil {
		t.Fatal(err)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./pkg/vram/ -run TestEstimateHeuristic -v`
Expected: FAIL — `undefined: EstimateHeuristic`.

- [ ] **Step 3: Implement the heuristic**

Create `pkg/vram/heuristic.go`:

```go
package vram

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Heuristic constants. Weights are typically already stored in their runtime
// dtype (fp16 safetensors), so VRAM ≈ on-disk weights × a small inflation for
// activations/working buffers, plus a fixed overhead for the CUDA context and
// auxiliary modules (VAE, text encoders). These are deliberately conservative
// seeds; the measured value (Task 4.1) corrects them after the first load.
const (
	heuristicWeightFactor   = 1.2
	heuristicFixedOverhead  = uint64(2) << 30 // 2 GiB
	heuristicMinimumEstimate = uint64(1) << 30 // never return less than 1 GiB for a GPU model
)

var weightExtensions = map[string]bool{
	".safetensors": true,
	".bin":         true,
	".pt":          true,
	".pth":         true,
	".ckpt":        true,
	".gguf":        true,
}

// EstimateHeuristic sizes a model from its on-disk weights when metadata-based
// estimation is unavailable. modelPath may be a single weight file or a model
// directory (diffusers pipeline); directories are walked and weight files
// summed. Returns 0 only when no weight bytes are found.
func EstimateHeuristic(modelPath string) uint64 {
	fi, err := os.Stat(modelPath)
	if err != nil {
		return 0
	}

	var weightBytes uint64
	if fi.IsDir() {
		_ = filepath.WalkDir(modelPath, func(p string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			if weightExtensions[strings.ToLower(filepath.Ext(p))] {
				if info, e := d.Info(); e == nil {
					weightBytes += uint64(info.Size())
				}
			}
			return nil
		})
	} else {
		weightBytes = uint64(fi.Size())
	}

	if weightBytes == 0 {
		return 0
	}
	est := uint64(float64(weightBytes)*heuristicWeightFactor) + heuristicFixedOverhead
	if est < heuristicMinimumEstimate {
		est = heuristicMinimumEstimate
	}
	return est
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./pkg/vram/ -run TestEstimateHeuristic -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pkg/vram/heuristic.go pkg/vram/heuristic_test.go
git commit -m "feat(vram): file-size heuristic estimator for unparseable models"
```

---

### Task 2.2: `ModelVRAMEstimate` model + accessors

**Files:**
- Modify: `core/services/nodes/registry.go` (model near `NodeGPU`; migration list; accessors near other Get/Upsert methods)
- Test: `core/services/nodes/registry_test.go`

- [ ] **Step 1: Write the failing test**

```go
	Describe("ModelVRAMEstimate cache", func() {
		It("upserts and reads back, measured overwrites heuristic", func() {
			ctx := context.Background()
			Expect(registry.UpsertModelVRAMEstimate(ctx, "sdxl", "diffusers", 9e9, "heuristic", 1)).To(Succeed())
			got, err := registry.GetModelVRAMEstimate(ctx, "sdxl")
			Expect(err).ToNot(HaveOccurred())
			Expect(got.VRAMBytes).To(Equal(uint64(9e9)))
			Expect(got.Source).To(Equal("heuristic"))

			Expect(registry.UpsertModelVRAMEstimate(ctx, "sdxl", "diffusers", 7_500_000_000, "measured", 1)).To(Succeed())
			got, err = registry.GetModelVRAMEstimate(ctx, "sdxl")
			Expect(err).ToNot(HaveOccurred())
			Expect(got.VRAMBytes).To(Equal(uint64(7_500_000_000)))
			Expect(got.Source).To(Equal("measured"))
		})
		It("returns gorm.ErrRecordNotFound for an unknown model", func() {
			_, err := registry.GetModelVRAMEstimate(context.Background(), "nope")
			Expect(err).To(MatchError(gorm.ErrRecordNotFound))
		})
	})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: FAIL — `undefined: registry.UpsertModelVRAMEstimate`.

- [ ] **Step 3: Add model, migration, accessors**

In `registry.go`, add the model:

```go
// ModelVRAMEstimate caches the VRAM footprint of a model, keyed by model name.
// source is "heuristic" (file-size seed) or "measured" (observed after a load).
// A measured row always overwrites a heuristic one; this drives auto multi-GPU
// span decisions in the scheduler.
type ModelVRAMEstimate struct {
	ModelName        string    `gorm:"primaryKey;size:255" json:"model_name"`
	Backend          string    `gorm:"size:128" json:"backend"`
	VRAMBytes        uint64    `gorm:"column:vram_bytes" json:"vram_bytes"`
	Source           string    `gorm:"size:16" json:"source"`
	GPUCountObserved int       `gorm:"column:gpu_count_observed;default:1" json:"gpu_count_observed"`
	UpdatedAt        time.Time `json:"updated_at"`
}

func (ModelVRAMEstimate) TableName() string { return "model_vram_estimates" }
```

Add `&ModelVRAMEstimate{}` to the `AutoMigrate` list.

Add accessors:

```go
// GetModelVRAMEstimate returns the cached estimate for a model, or
// gorm.ErrRecordNotFound if none exists.
func (r *NodeRegistry) GetModelVRAMEstimate(ctx context.Context, modelName string) (*ModelVRAMEstimate, error) {
	var e ModelVRAMEstimate
	if err := r.db.WithContext(ctx).Where("model_name = ?", modelName).First(&e).Error; err != nil {
		return nil, err
	}
	return &e, nil
}

// UpsertModelVRAMEstimate writes (or overwrites) the cached estimate. A
// "measured" source always wins over a "heuristic" one; a fresh "heuristic"
// must NOT clobber an existing "measured" value.
func (r *NodeRegistry) UpsertModelVRAMEstimate(ctx context.Context, modelName, backend string, bytes uint64, source string, gpuCount int) error {
	if source == "heuristic" {
		// Don't downgrade a measured row back to a heuristic seed.
		if existing, err := r.GetModelVRAMEstimate(ctx, modelName); err == nil && existing.Source == "measured" {
			return nil
		}
	}
	row := ModelVRAMEstimate{ModelName: modelName, Backend: backend, VRAMBytes: bytes, Source: source, GPUCountObserved: gpuCount, UpdatedAt: time.Now()}
	return r.db.WithContext(ctx).Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "model_name"}},
		DoUpdates: clause.AssignmentColumns([]string{"backend", "vram_bytes", "source", "gpu_count_observed", "updated_at"}),
	}).Create(&row).Error
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/registry.go core/services/nodes/registry_test.go
git commit -m "feat(nodes): ModelVRAMEstimate cache with measured-wins-over-heuristic"
```

---

### Task 2.3: `estimateModelVRAM` lookup order

**Files:**
- Modify: `core/services/nodes/router.go` (`estimateModelVRAM` ~line 906)
- Test: `core/services/nodes/router_test.go`

Note: `scheduleNewModel` calls `r.vramEstimator(ctx, modelOpts)`. We change the production `estimateModelVRAM` so it consults the cache first, then metadata, then the heuristic, and seeds the cache with the heuristic. It needs `modelOpts.Model`/`ModelFile` and the model name (`trackingKey`). The estimator signature only has `*pb.ModelOptions`; the model name is `opts.Model`. Use `opts.Model` as the cache key (matches `trackingKey`/model id used elsewhere).

- [ ] **Step 1: Write the failing test**

In `router_test.go`, add (the suite already builds a `SmartRouter` with a fake registry — reuse that harness; assert the lookup precedence via a fake that returns a measured row):

```go
	It("estimateModelVRAM prefers a measured cache row over heuristic", func() {
		// fakeModelRouter is the existing test double; extend it to return a
		// measured estimate for model "m".
		r := newRouterWithFakes() // existing helper in this suite
		r.registry.(*fakeModelRouter).vramEstimate = &ModelVRAMEstimate{ModelName: "m", VRAMBytes: 7e9, Source: "measured"}
		got := r.estimateModelVRAM(context.Background(), &pb.ModelOptions{Model: "m"})
		Expect(got).To(Equal(uint64(7e9)))
	})
```

If the existing fake registry can't carry `vramEstimate`/`GetModelVRAMEstimate`, add the method to the fake and a settable field (mirror how other fake methods are stubbed in `router_test.go`).

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: FAIL — estimator ignores the cache (returns 0 / metadata value).

- [ ] **Step 3: Rewrite `estimateModelVRAM`**

Replace the body of `estimateModelVRAM` (router.go:906) with:

```go
func (r *SmartRouter) estimateModelVRAM(ctx context.Context, opts *pb.ModelOptions) uint64 {
	estCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	// 1. Measured/cached estimate wins (set after a real load, see Task 4.1).
	if opts.Model != "" {
		if cached, err := r.registry.GetModelVRAMEstimate(ctx, opts.Model); err == nil && cached.VRAMBytes > 0 {
			return cached.VRAMBytes
		}
	}

	ctxSize := uint32(opts.ContextSize)
	if ctxSize == 0 {
		ctxSize = 8192
	}

	// 2. Metadata estimate (GGUF / HF repo) for parseable models.
	input := vram.ModelEstimateInput{Options: vram.EstimateOptions{GPULayers: int(opts.NGPULayers)}}
	if opts.ModelFile != "" {
		if _, err := os.Stat(opts.ModelFile); err == nil {
			input.Files = append(input.Files, vram.FileInput{URI: opts.ModelFile, Size: 0})
		}
	}
	if opts.Model != "" {
		if repoID, ok := vram.ExtractHFRepoID(opts.Model); ok {
			input.HFRepo = repoID
		}
	}
	if len(input.Files) > 0 || input.HFRepo != "" || input.Size != "" {
		if result, err := vram.EstimateModelMultiContext(estCtx, input, []uint32{ctxSize}); err == nil {
			if v := result.VRAMForContext(ctxSize); v > 0 {
				return v
			}
		}
	}

	// 3. File-size heuristic for unparseable models (diffusers/SDXL). Seed the
	//    cache so subsequent placements are consistent until a measured value
	//    replaces it.
	heur := vram.EstimateHeuristic(opts.ModelFile)
	if heur > 0 && opts.Model != "" {
		if err := r.registry.UpsertModelVRAMEstimate(ctx, opts.Model, "", heur, "heuristic", 1); err != nil {
			xlog.Debug("failed to seed heuristic VRAM estimate", "model", opts.Model, "error", err)
		}
	}
	return heur
}
```

Add `GetModelVRAMEstimate` and `UpsertModelVRAMEstimate` to the registry interface used by `SmartRouter` (search the interface in `router.go`/`interfaces.go` that lists `ReserveVRAM`, `FindNodeWithVRAM`, etc., and add the two method signatures). Update the test fake accordingly.

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/router.go core/services/nodes/router_test.go core/services/nodes/interfaces.go
git commit -m "feat(nodes): estimate VRAM via measured cache, metadata, then heuristic"
```

**Phase 2 gate:** `go build ./...` and `go test ./pkg/vram/... ./core/services/nodes/...` green. After this phase, diffusers models size non-zero, so the existing node-level VRAM filter engages (a partial improvement); Phase 3 makes it per-GPU and correct.

---

# Phase 3 — Per-GPU placement + enforcement

Goal: scheduler picks a specific GPU set on a node (best-fit / fewest-GPU spanning), reserves per-GPU, pins the worker process via `CUDA_VISIBLE_DEVICES`, and sets the multi-GPU knob to the set size. This is the behavioral fix for the incident.

**Pre-Phase-3 verification (do this first).** Confirm what `estimateModelVRAM` actually resolves for a real diffusers model in distributed mode — specifically whether `opts.ModelFile` is a path the **frontend** pod can `os.Stat` (diffusers weights may live only on the worker's NFS `/backends/...`). Add a temporary debug log in `estimateModelVRAM` (or inspect via a unit/integration check) for one of the cluster's SDXL models and confirm the returned estimate is non-zero. **This is not a blocker** — the `estimate == 0` liveness fallback (Tasks 3.1/3.4) keeps loads working and still pins to one GPU even when the estimate is unknown — but knowing the answer tells you whether spanning (which needs a real size) will ever trigger for diffusers, or whether you also need a worker-side estimate source. Record the finding inline in the plan before proceeding.

### Task 3.1: `planGPUSet` pure placement function

**Files:**
- Create: `core/services/nodes/gpuplan.go`
- Test: `core/services/nodes/gpuplan_test.go`

- [ ] **Step 1: Write the failing test**

Create `core/services/nodes/gpuplan_test.go`:

```go
package nodes

import "testing"

func gpu(idx int, free uint64) NodeGPU { return NodeGPU{GPUIndex: idx, FreeVRAM: free, TotalVRAM: 12 << 30} }

func TestPlanGPUSet(t *testing.T) {
	const GiB = uint64(1) << 30
	cases := []struct {
		name     string
		gpus     []NodeGPU
		estimate uint64
		want     []int
		wantOK   bool
	}{
		{"fits one, best-fit picks smallest-fitting", []NodeGPU{gpu(0, 11 * GiB), gpu(1, 7 * GiB)}, 6 * GiB, []int{1}, true},
		{"one full one free lands on free", []NodeGPU{gpu(0, 200 * 1 << 20), gpu(1, 11 * GiB)}, 7 * GiB, []int{1}, true},
		{"needs spanning, fewest GPUs", []NodeGPU{gpu(0, 8 * GiB), gpu(1, 8 * GiB), gpu(2, 8 * GiB)}, 15 * GiB, []int{0, 1}, true},
		{"does not fit anywhere", []NodeGPU{gpu(0, 4 * GiB), gpu(1, 4 * GiB)}, 20 * GiB, nil, false},
		{"reserved reduces effective free", []NodeGPU{{GPUIndex: 0, FreeVRAM: 11 * GiB, ReservedVRAM: 6 * GiB}}, 6 * GiB, nil, false},
		{"zero estimate does not fit (liveness handled by caller)", []NodeGPU{gpu(0, 11 * GiB)}, 0, nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := planGPUSet(tc.gpus, tc.estimate)
			if ok != tc.wantOK {
				t.Fatalf("ok=%v want %v", ok, tc.wantOK)
			}
			if !intSliceEqual(got, tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}

func TestLargestFreeGPU(t *testing.T) {
	const GiB = uint64(1) << 30
	idx, ok := largestFreeGPU([]NodeGPU{gpu(0, 2 * GiB), gpu(1, 9 * GiB), gpu(2, 5 * GiB)})
	if !ok || idx != 1 {
		t.Fatalf("largestFreeGPU = (%d,%v), want (1,true)", idx, ok)
	}
	if _, ok := largestFreeGPU(nil); ok {
		t.Fatalf("largestFreeGPU(nil) ok=true, want false")
	}
	if _, ok := largestFreeGPU([]NodeGPU{{GPUIndex: 0, FreeVRAM: 1 * GiB, ReservedVRAM: 1 * GiB}}); ok {
		t.Fatalf("largestFreeGPU with no effective-free ok=true, want false")
	}
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestPlanGPUSet -v`
Expected: FAIL — `undefined: planGPUSet`.

- [ ] **Step 3: Implement `planGPUSet`**

Create `core/services/nodes/gpuplan.go`:

```go
package nodes

import "sort"

// effectiveFree is a GPU's free VRAM minus its in-tick soft reservation.
func effectiveFree(g NodeGPU) uint64 {
	if g.ReservedVRAM >= g.FreeVRAM {
		return 0
	}
	return g.FreeVRAM - g.ReservedVRAM
}

// planGPUSet decides which physical GPU indices on a single node should host a
// model needing `estimate` bytes. Policy:
//   - Best-fit on ONE GPU: the smallest effective-free GPU that still fits
//     (preserves whole GPUs for models that actually need them).
//   - Else span the FEWEST GPUs (largest effective-free first) whose combined
//     effective-free covers the estimate.
//   - Returns (nil, false) when the node cannot fit the model.
// Returned indices are sorted ascending for deterministic assignment.
func planGPUSet(gpus []NodeGPU, estimate uint64) ([]int, bool) {
	if estimate == 0 || len(gpus) == 0 {
		return nil, false
	}

	// 1. Best-fit single GPU.
	bestIdx := -1
	var bestFree uint64
	for _, g := range gpus {
		ef := effectiveFree(g)
		if ef >= estimate && (bestIdx == -1 || ef < bestFree) {
			bestIdx = g.GPUIndex
			bestFree = ef
		}
	}
	if bestIdx != -1 {
		return []int{bestIdx}, true
	}

	// 2. Span fewest GPUs: take largest effective-free first until covered.
	sorted := make([]NodeGPU, len(gpus))
	copy(sorted, gpus)
	sort.Slice(sorted, func(i, j int) bool { return effectiveFree(sorted[i]) > effectiveFree(sorted[j]) })

	var acc uint64
	var chosen []int
	for _, g := range sorted {
		ef := effectiveFree(g)
		if ef == 0 {
			continue
		}
		acc += ef
		chosen = append(chosen, g.GPUIndex)
		if acc >= estimate {
			sort.Ints(chosen)
			return chosen, true
		}
	}
	return nil, false
}

// largestFreeGPU returns the index of the GPU with the most effective-free
// VRAM (>0). Used as the liveness fallback when the model's VRAM estimate is
// unknown (0): we can't size it, but we can still pin it to ONE GPU (fixing the
// cuda:0 pile-up) instead of failing the load. Returns (_, false) when no GPU
// has any effective-free VRAM.
func largestFreeGPU(gpus []NodeGPU) (int, bool) {
	bestIdx := -1
	var bestFree uint64
	for _, g := range gpus {
		ef := effectiveFree(g)
		if ef > 0 && (bestIdx == -1 || ef > bestFree) {
			bestIdx = g.GPUIndex
			bestFree = ef
		}
	}
	if bestIdx == -1 {
		return 0, false
	}
	return bestIdx, true
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./core/services/nodes/ -run TestPlanGPUSet -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/gpuplan.go core/services/nodes/gpuplan_test.go
git commit -m "feat(nodes): planGPUSet best-fit single-GPU / fewest-GPU spanning"
```

---

### Task 3.2: `BackendInstallRequest.GPUIndices` + `InstallBackend` signature

**Files:**
- Modify: `core/services/messaging/subjects.go` (`BackendInstallRequest`)
- Modify: `core/services/nodes/unloader.go` (`NodeCommandSender.InstallBackend` + impl)
- Modify: `core/services/nodes/router.go` (`installBackendOnNode` ~line 957 passes new arg)
- Test: covered by build + Task 3.4 integration; add a marshalling test.

- [ ] **Step 1: Write the failing test**

Create `core/services/messaging/subjects_test.go` (or append):

```go
package messaging

import (
	"encoding/json"
	"testing"
)

func TestBackendInstallRequestGPUIndices(t *testing.T) {
	req := BackendInstallRequest{Backend: "diffusers", ModelID: "m", GPUIndices: []int{1, 2}}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	var got BackendInstallRequest
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if len(got.GPUIndices) != 2 || got.GPUIndices[0] != 1 || got.GPUIndices[1] != 2 {
		t.Fatalf("GPUIndices round-trip failed: %#v", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/messaging/ -run TestBackendInstallRequestGPUIndices -v`
Expected: FAIL — `unknown field GPUIndices`.

- [ ] **Step 3: Add the field and thread the arg**

In `subjects.go`, add to `BackendInstallRequest`:

```go
	// GPUIndices are the physical GPU indices the scheduler assigned to this
	// backend. The worker sets CUDA_VISIBLE_DEVICES to these so the process
	// only sees its assigned cards. Empty means "no pinning" (CPU / single-GPU
	// legacy behavior).
	GPUIndices []int `json:"gpu_indices,omitempty"`
```

In `unloader.go`, change the interface method signature to add `gpuIndices []int` and set it on the published request:

```go
	InstallBackend(nodeID, backendType, modelID, galleriesJSON, uri, name, alias string, replicaIndex int, opID string, gpuIndices []int, onProgress func(messaging.BackendInstallProgressEvent)) (*messaging.BackendInstallReply, error)
```

In the `RemoteUnloaderAdapter.InstallBackend` implementation, add `gpuIndices []int` to the params and set `GPUIndices: gpuIndices` in the `messaging.BackendInstallRequest{...}` literal.

In `router.go` `installBackendOnNode` (~957), add a `gpuIndices []int` parameter and pass it through to `r.unloader.InstallBackend(...)`; update the singleflight key to include the indices (so a re-install with a different GPU set isn't coalesced):

```go
	key := fmt.Sprintf("%s|%s|%s|%d|%v", node.ID, backendType, modelID, replicaIndex, gpuIndices)
	...
		reply, err := r.unloader.InstallBackend(node.ID, backendType, modelID, r.galleriesJSON, "", "", "", replicaIndex, "", gpuIndices, nil)
```

Find every other caller/implementer of `InstallBackend` (admin endpoint `InstallBackendOnNodeEndpoint`, any tests/fakes) and update them to pass `nil` for `gpuIndices` (admin manual install = no auto-pin) so the build stays green.

- [ ] **Step 4: Run test + build**

Run: `go test ./core/services/messaging/ -run TestBackendInstallRequestGPUIndices -v` → PASS
Run: `go build ./...` → success (fix any remaining `InstallBackend` callers).

- [ ] **Step 5: Commit**

```bash
git add core/services/messaging/subjects.go core/services/messaging/subjects_test.go core/services/nodes/unloader.go core/services/nodes/router.go core/http/endpoints/localai/
git commit -m "feat(nodes): carry assigned GPU indices through backend.install"
```

---

### Task 3.3: Worker sets `CUDA_VISIBLE_DEVICES` from `GPUIndices`

**Files:**
- Modify: `pkg/model/process.go` (`StartProcess`/`startProcess` ~line 132)
- Modify: `core/services/worker/supervisor.go` (`startBackend` ~line 64)
- Modify: `core/services/worker/install.go` (`installBackend` → `startBackend` call ~line 161; pass `req.GPUIndices`)
- Modify: `core/services/worker/lifecycle.go` (`handleBackendInstall` passes the request through — already does)
- Test: `pkg/model/process_test.go` (env-construction helper) or a focused unit test.

- [ ] **Step 1: Write the failing test**

Create `core/services/worker/cuda_env_test.go`:

```go
package worker

import "testing"

func TestCudaVisibleDevicesEnv(t *testing.T) {
	if got := cudaVisibleDevicesEnv(nil); got != "" {
		t.Fatalf("nil indices: got %q want empty", got)
	}
	if got := cudaVisibleDevicesEnv([]int{1}); got != "CUDA_VISIBLE_DEVICES=1" {
		t.Fatalf("single: got %q", got)
	}
	if got := cudaVisibleDevicesEnv([]int{2, 0}); got != "CUDA_VISIBLE_DEVICES=2,0" {
		t.Fatalf("multi (order preserved): got %q", got)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/worker/ -run TestCudaVisibleDevicesEnv -v`
Expected: FAIL — `undefined: cudaVisibleDevicesEnv`.

- [ ] **Step 3: Implement env injection end-to-end**

In `supervisor.go`, add the helper and thread indices into `startBackend`:

```go
// cudaVisibleDevicesEnv builds the CUDA_VISIBLE_DEVICES assignment for a backend
// process, or "" when no GPUs were assigned (CPU / legacy single-GPU).
func cudaVisibleDevicesEnv(gpuIndices []int) string {
	if len(gpuIndices) == 0 {
		return ""
	}
	parts := make([]string, len(gpuIndices))
	for i, idx := range gpuIndices {
		parts[i] = strconv.Itoa(idx)
	}
	return "CUDA_VISIBLE_DEVICES=" + strings.Join(parts, ",")
}
```

Change `startBackend`'s signature to `func (s *backendSupervisor) startBackend(backend, backendPath string, gpuIndices []int) (string, error)` and pass an env override into the process start:

```go
	proc, err := s.ml.StartProcess(backendPath, backend, bindAddr, cudaVisibleDevicesEnv(gpuIndices))
```

(If `StartProcess`'s variadic `args ...string` would swallow the env string, add an explicit `extraEnv []string` parameter instead — see below.)

In `pkg/model/process.go`, change `StartProcess`/`startProcess` to accept an extra-env slice and apply it on top of `os.Environ()`:

```go
func (ml *ModelLoader) StartProcess(grpcProcess, id, serverAddress string, extraEnv ...string) (*process.Process, error) {
	return ml.startProcess(grpcProcess, id, serverAddress, extraEnv)
}

func (ml *ModelLoader) startProcess(grpcProcess, id, serverAddress string, extraEnv []string, args ...string) (*process.Process, error) {
	...
	env := os.Environ()
	for _, e := range extraEnv {
		if e != "" {
			env = append(env, e) // later entries win in exec; CUDA_VISIBLE_DEVICES override
		}
	}
	grpcControlProcess := process.New(
		process.WithTemporaryStateDir(),
		process.WithName(filepath.Base(grpcProcess)),
		process.WithArgs(append(args, []string{"--addr", serverAddress}...)...),
		process.WithEnvironment(env...),
		process.WithWorkDir(workDir),
	)
	...
}
```

(Adjust the existing `serve-backend` CLI caller of `StartProcess` to pass no extra env.)

In `install.go`, pass the request's indices:

```go
	return s.startBackend(processKey, backendPath, req.GPUIndices)
```

Note: `installBackend` takes `req messaging.BackendInstallRequest`, so `req.GPUIndices` is available. Ensure `int32` vs `int`: `GPUIndices` is `[]int` on the wire, so no conversion needed.

Ensure `supervisor.go` imports `strconv` and `strings` (it already imports both).

- [ ] **Step 4: Run test + build**

Run: `go test ./core/services/worker/ -run TestCudaVisibleDevicesEnv -v` → PASS
Run: `go build ./...` → success.

- [ ] **Step 5: Commit**

```bash
git add core/services/worker/supervisor.go core/services/worker/install.go pkg/model/process.go core/services/worker/cuda_env_test.go
git commit -m "feat(worker): pin backend process to assigned GPUs via CUDA_VISIBLE_DEVICES"
```

---

### Task 3.4: `scheduleNewModel` uses per-GPU placement; set multi-GPU knob

**Files:**
- Modify: `core/services/nodes/registry.go` (add `NodeModel.GPUIndices` column)
- Modify: `core/services/nodes/router.go` (`scheduleNewModel` body ~687-903; `scheduleAndLoad` ~172; add `applyGPUSet`)
- Test: `core/services/nodes/router_test.go`

- [ ] **Step 1: Write the failing test**

Add to `router_test.go` a test that, given a node whose `node_gpus` has one full + one free GPU and a model estimate that fits one GPU, `scheduleNewModel` returns the free GPU's index and installs with that index. Use the existing fake harness; assert the `GPUIndices` captured by the fake `InstallBackend`:

```go
	It("places a single-GPU model on the free GPU, not the full one", func() {
		r := newRouterWithFakes()
		f := r.registry.(*fakeModelRouter)
		f.node = &BackendNode{ID: "n", Name: "noctis", Status: StatusHealthy, NodeType: NodeTypeBackend}
		f.nodeGPUs = []NodeGPU{{NodeID: "n", GPUIndex: 0, FreeVRAM: 200 << 20}, {NodeID: "n", GPUIndex: 1, FreeVRAM: 11 << 30}}
		f.vramEstimate = &ModelVRAMEstimate{ModelName: "m", VRAMBytes: 7 << 30, Source: "measured"}

		_, _, _, err := r.scheduleNewModel(context.Background(), "diffusers", "m", &pb.ModelOptions{Model: "m"})
		Expect(err).ToNot(HaveOccurred())
		Expect(f.lastInstallGPUIndices).To(Equal([]int{1}))
	})
```

Extend `fakeModelRouter` with `nodeGPUs []NodeGPU`, `NodeGPUs(...)` returning it, `lastInstallGPUIndices []int` captured in the fake `InstallBackend`, and the per-GPU reserve methods as no-ops returning nil.

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: FAIL — scheduler ignores per-GPU data / doesn't pass indices.

- [ ] **Step 3: Implement**

(a) Add column to `NodeModel` in `registry.go`:

```go
	GPUIndices    string    `gorm:"column:gpu_indices;size:64" json:"gpu_indices,omitempty"` // CSV of assigned physical GPU indices
```

Extend `SetNodeModel` to accept and persist `gpuIndices string` (add the param; set it in the `Assign(map[string]any{...})`), and update all callers.

(b) In `router.go`, add the helpers:

```go
// gpuIndicesCSV renders assigned indices for storage on the NodeModel row.
func gpuIndicesCSV(idx []int) string {
	parts := make([]string, len(idx))
	for i, v := range idx {
		parts[i] = strconv.Itoa(v)
	}
	return strings.Join(parts, ",")
}

// applyGPUSet sets the backend's multi-GPU knob to match the assigned GPU
// count. CUDA_VISIBLE_DEVICES already restricts the process to these cards, so
// the backend sees them as 0..K-1. K=1 leaves single-device behavior. An
// explicit TensorParallelSize already larger than K in the config is respected
// (the operator asked for more parallelism).
func applyGPUSet(opts *pb.ModelOptions, k int) {
	if opts == nil || k < 1 {
		return
	}
	if int(opts.TensorParallelSize) < k {
		opts.TensorParallelSize = int32(k)
	}
	if k > 1 && opts.TensorSplit == "" {
		even := make([]string, k)
		for i := range even {
			even[i] = "1"
		}
		opts.TensorSplit = strings.Join(even, ",")
	}
}
```

(c) Rewrite the placement section of `scheduleNewModel`. Replace the `estimatedVRAM>0 ? FindNodeWithVRAM... : FindIdleNode...` block (lines ~734-763) and the node-level reservation (~876-899) with per-GPU selection. The new flow, after computing `estimatedVRAM` and `candidateNodeIDs`:

```go
	// Per-GPU placement: pick a node AND a GPU set on it. Iterate candidate
	// nodes ordered by most effective-free VRAM; ask planGPUSet to fit the
	// model on one GPU (best-fit) or span the fewest GPUs.
	var node *BackendNode
	var gpuSet []int
	sawPerGPUNode := false
	nodesOrdered, nodesErr := r.registry.CandidateNodesByFreeVRAM(ctx, candidateNodeIDs)
	if nodesErr == nil {
		for i := range nodesOrdered {
			cand := &nodesOrdered[i]
			gpus, gerr := r.registry.NodeGPUs(ctx, cand.ID)
			if gerr != nil || len(gpus) == 0 {
				continue // node hasn't reported per-GPU data yet (e.g. old worker)
			}
			sawPerGPUNode = true
			if estimatedVRAM > 0 {
				if set, ok := planGPUSet(gpus, estimatedVRAM); ok {
					node = cand
					gpuSet = set
					break
				}
			} else {
				// Unknown estimate (e.g. diffusers files not stat-able on the
				// frontend): preserve liveness — pin to the single largest-free
				// GPU on the most-free node. No reservation (we don't know the
				// size). This still fixes the cuda:0 pile-up. nodesOrdered is
				// sorted by node free VRAM, so the first node we reach is best.
				if gi, ok := largestFreeGPU(gpus); ok {
					node = cand
					gpuSet = []int{gi}
					break
				}
			}
		}
	}

	// Mixed-version safety (spec §8): if NO candidate node reported per-GPU
	// data, fall back to the legacy node-level VRAM path. gpuSet stays empty,
	// so the worker gets no CUDA_VISIBLE_DEVICES pin (today's behavior).
	if node == nil && !sawPerGPUNode && estimatedVRAM > 0 {
		if candidateNodeIDs != nil {
			node, _ = r.registry.FindNodeWithVRAMFromSet(ctx, estimatedVRAM, candidateNodeIDs)
		} else {
			node, _ = r.registry.FindNodeWithVRAM(ctx, estimatedVRAM)
		}
	}

	// Eviction fallback when no node+GPU-set fit. Mirror the existing loop
	// (evictionBusyRetries / postEvictPoll constants stay), but evaluate the
	// freed node with planGPUSet instead of FindNodeWithVRAMFromSet.
	if node == nil {
		const (
			evictionBusyRetries = 6
			evictionBusyDelay   = 500 * time.Millisecond
		)
		for outer := 0; outer < evictionBusyRetries; outer++ {
			evictedNode, evictErr := r.evictLRUAndFreeNode(ctx)
			if evictErr != nil {
				if !errors.Is(evictErr, ErrEvictionBusy) {
					return nil, "", 0, nil, fmt.Errorf("no node fits and eviction failed: %w", evictErr)
				}
				select {
				case <-time.After(evictionBusyDelay):
					continue
				case <-ctx.Done():
					return nil, "", 0, nil, fmt.Errorf("eviction interrupted: %w", ctx.Err())
				}
			}
			// Poll the freed node's per-GPU data to let the worker's heartbeat
			// refresh free_vram after the physical unload, then re-plan.
			for poll := 0; poll < 6; poll++ {
				gpus, gerr := r.registry.NodeGPUs(ctx, evictedNode.ID)
				if gerr == nil && len(gpus) > 0 {
					if estimatedVRAM > 0 {
						if set, ok := planGPUSet(gpus, estimatedVRAM); ok {
							node, gpuSet = evictedNode, set
							break
						}
					} else if gi, ok := largestFreeGPU(gpus); ok {
						node, gpuSet = evictedNode, []int{gi}
						break
					}
				}
				select {
				case <-time.After(500 * time.Millisecond):
				case <-ctx.Done():
					return nil, "", 0, nil, fmt.Errorf("eviction VRAM-recheck interrupted: %w", ctx.Err())
				}
			}
			if node != nil {
				break
			}
		}
	}
```

Add `CandidateNodesByFreeVRAM(ctx, ids)` to the `NodeRegistry` AND to the `SmartRouter` registry interface (the one already declaring `FindNodeWithVRAM`, `ReserveVRAM`, `NodeGPUs`, etc.) and to the test fake. Implementation: healthy backend nodes (optionally filtered to `ids`) ordered by `(available_vram - reserved_vram) DESC`:

```go
func (r *NodeRegistry) CandidateNodesByFreeVRAM(ctx context.Context, nodeIDs []string) ([]BackendNode, error) {
	q := r.db.WithContext(ctx).
		Where("status = ? AND node_type = ?", StatusHealthy, NodeTypeBackend)
	if len(nodeIDs) > 0 {
		q = q.Where("id IN ?", nodeIDs)
	}
	var nodes []BackendNode
	err := q.Order("(available_vram - reserved_vram) DESC").Find(&nodes).Error
	return nodes, err
}
```

(d) Per-GPU reservation in place of the node-level one (~876):

```go
	reservedGPUs := []int{}
	if len(gpuSet) > 0 && estimatedVRAM > 0 {
		per := estimatedVRAM / uint64(len(gpuSet))
		for _, gi := range gpuSet {
			if err := r.registry.ReserveVRAMOnGPU(ctx, node.ID, gi, per); err != nil {
				xlog.Warn("per-GPU reservation failed, proceeding", "node", node.Name, "gpu", gi, "error", err)
			} else {
				reservedGPUs = append(reservedGPUs, gi)
			}
		}
	}
```

(e) Pass `gpuSet` to `installBackendOnNode(ctx, node, backendType, modelID, replicaIdx, gpuSet)`; on install error, release per-GPU reservations:

```go
	if installErr != nil {
		per := estimatedVRAM / uint64(max(1, len(gpuSet)))
		for _, gi := range reservedGPUs {
			_ = r.registry.ReleaseVRAMOnGPU(ctx, node.ID, gi, per)
		}
		return nil, "", 0, fmt.Errorf("installing backend on node %s: %w", node.Name, installErr)
	}
```

(f) Change `scheduleNewModel`'s return to include `gpuSet` (return `(*BackendNode, string, int, []int, error)`), and in `scheduleAndLoad` (~172): after a successful schedule, call `applyGPUSet(loadOpts, len(gpuSet))` before `LoadModel`, and pass `gpuIndicesCSV(gpuSet)` to `SetNodeModel`.

- [ ] **Step 4: Run test + build**

Run: `go test ./core/services/nodes/ -run TestNodes -v` → PASS
Run: `go build ./...` → success.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/router.go core/services/nodes/registry.go core/services/nodes/router_test.go
git commit -m "feat(nodes): per-GPU placement, reservation, and multi-GPU knob"
```

---

### Task 3.5: Bounded retries + terminal error (no more retry-forever)

**Files:**
- Modify: `core/services/nodes/router.go` (`scheduleNewModel` no-fit terminal path)
- Test: `core/services/nodes/router_test.go`

- [ ] **Step 1: Write the failing test**

```go
	It("returns a clear terminal error when no GPU set on any node fits", func() {
		r := newRouterWithFakes()
		f := r.registry.(*fakeModelRouter)
		f.node = &BackendNode{ID: "n", Name: "small", Status: StatusHealthy, NodeType: NodeTypeBackend}
		f.nodeGPUs = []NodeGPU{{NodeID: "n", GPUIndex: 0, FreeVRAM: 2 << 30}}
		f.vramEstimate = &ModelVRAMEstimate{ModelName: "big", VRAMBytes: 40 << 30, Source: "measured"}
		f.evictErr = ErrEvictionBusy // nothing to evict

		_, _, _, _, err := r.scheduleNewModel(context.Background(), "diffusers", "big", &pb.ModelOptions{Model: "big"})
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("more VRAM than any GPU set"))
	})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: FAIL — generic/other error.

- [ ] **Step 3: Implement the terminal error (only when we have a real estimate)**

In `scheduleNewModel`, after the eviction loop, distinguish the two no-node cases. A real estimate that fits nowhere is a definitive failure. An UNKNOWN estimate (0) must NOT hard-fail — fall back to the legacy VRAM-blind placement so liveness is preserved exactly as before this change (this also covers all-old-workers clusters where no per-GPU data exists):

```go
	if node == nil {
		if estimatedVRAM > 0 {
			return nil, "", 0, nil, fmt.Errorf(
				"model %s needs ~%s, more VRAM than any GPU set on any node can provide",
				modelID, vram.FormatBytes(estimatedVRAM))
		}
		// Unknown estimate: last-resort legacy placement (no GPU pin). Preserves
		// pre-change behavior rather than failing a load we can't size.
		if candidateNodeIDs != nil {
			node, _ = r.registry.FindIdleNodeFromSet(ctx, candidateNodeIDs)
			if node == nil {
				node, _ = r.registry.FindLeastLoadedNodeFromSet(ctx, candidateNodeIDs)
			}
		} else {
			node, _ = r.registry.FindIdleNode(ctx)
			if node == nil {
				node, _ = r.registry.FindLeastLoadedNode(ctx)
			}
		}
		if node == nil {
			return nil, "", 0, nil, fmt.Errorf("no healthy node available for model %s", modelID)
		}
		// gpuSet stays empty → no CUDA_VISIBLE_DEVICES pin (legacy behavior).
	}
```

This propagates a *real* over-capacity failure to the caller as a definitive error (the worker-side 7-minute readiness retry loop never starts because we never install), while never regressing a model that simply couldn't be sized.

Note: the `largestFreeGPU` liveness path in Task 3.4 already pins unknown-estimate models to one GPU on any node with per-GPU data; this legacy fallback only triggers when *no* node reported per-GPU data at all (pure mixed-version / detection-blip case).

- [ ] **Step 4: Run test + build**

Run: `go test ./core/services/nodes/ -run TestNodes -v` → PASS
Run: `go build ./...` → success.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/router.go core/services/nodes/router_test.go
git commit -m "feat(nodes): terminal error when no GPU set fits, instead of retry loop"
```

**Phase 3 gate:** `go build ./...` and full `go test ./core/services/... ./pkg/...` green.

---

# Phase 4 — Measurement & resilience

Goal: correct the heuristic with a measured per-GPU delta after a successful load; bump the estimate and release reservations on failure.

### Task 4.1: Measure per-GPU delta after a successful load

**Files:**
- Modify: `core/services/nodes/router.go` (`scheduleAndLoad` after `LoadModel` success ~205)
- Test: `core/services/nodes/router_test.go`

Mechanism: capture each assigned GPU's `free_vram` from `node_gpus` just before install (we already read `NodeGPUs` in Task 3.4 — stash the pre-install free for the assigned indices), and after `LoadModel` success poll `NodeGPUs` for up to ~2 heartbeat intervals; the drop in free on the assigned GPUs is the footprint. Record only when no other `loading`/concurrent install touched those GPUs in the window (best-effort: check the assigned GPUs had no other NodeModel transition; if uncertain, skip and log).

- [ ] **Step 1: Write the failing test**

```go
	It("records a measured estimate from the per-GPU free delta after load", func() {
		r := newRouterWithFakes()
		f := r.registry.(*fakeModelRouter)
		// Pre-install free, then post-load free (8 GiB lower on the assigned GPU).
		f.nodeGPUsSeq = [][]NodeGPU{
			{{NodeID: "n", GPUIndex: 0, FreeVRAM: 12 << 30}},                 // pre
			{{NodeID: "n", GPUIndex: 0, FreeVRAM: 4 << 30}},                  // post
		}
		f.node = &BackendNode{ID: "n", Name: "noctis", Status: StatusHealthy, NodeType: NodeTypeBackend}
		f.vramEstimate = &ModelVRAMEstimate{ModelName: "m", VRAMBytes: 6 << 30, Source: "heuristic"}

		_, err := r.scheduleAndLoad(context.Background(), "diffusers", "m", "m", &pb.ModelOptions{Model: "m"}, false, 0)
		Expect(err).ToNot(HaveOccurred())
		Expect(f.lastUpsert.Source).To(Equal("measured"))
		Expect(f.lastUpsert.VRAMBytes).To(BeNumerically("~", uint64(8<<30), uint64(512<<20)))
	})
```

Extend `fakeModelRouter` with `nodeGPUsSeq` (returns successive snapshots) and `lastUpsert ModelVRAMEstimate`.

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: FAIL — no measured upsert occurs.

- [ ] **Step 3: Implement measurement**

In `scheduleAndLoad`, after the `SetNodeModel` success and only when `len(gpuSet) > 0`:

```go
	// Measure the real per-GPU footprint to correct the heuristic. Best-effort
	// and skipped when it can't be cleanly attributed.
	if len(result.GPUSet) > 0 {
		go r.recordMeasuredVRAM(context.Background(), node.ID, trackingKey, backendType, result.GPUSet, preInstallFree)
	}
```

Add:

```go
// recordMeasuredVRAM polls per-GPU free after a load and caches the observed
// footprint (sum of free drop on the assigned GPUs). preFree maps gpuIndex ->
// free bytes captured before install. Best-effort; logs and returns on noise.
func (r *SmartRouter) recordMeasuredVRAM(ctx context.Context, nodeID, modelName, backend string, gpuSet []int, preFree map[int]uint64) {
	const settleAttempts = 3
	const settleDelay = 6 * time.Second // ~heartbeat interval; tune to HeartbeatInterval
	for attempt := 0; attempt < settleAttempts; attempt++ {
		select {
		case <-time.After(settleDelay):
		case <-ctx.Done():
			return
		}
		gpus, err := r.registry.NodeGPUs(ctx, nodeID)
		if err != nil {
			continue
		}
		byIdx := map[int]uint64{}
		for _, g := range gpus {
			byIdx[g.GPUIndex] = g.FreeVRAM
		}
		var footprint uint64
		ok := true
		for _, gi := range gpuSet {
			pre, havePre := preFree[gi]
			post, havePost := byIdx[gi]
			if !havePre || !havePost || post >= pre {
				ok = false
				break
			}
			footprint += pre - post
		}
		if ok && footprint > 0 {
			if err := r.registry.UpsertModelVRAMEstimate(ctx, modelName, backend, footprint, "measured", len(gpuSet)); err != nil {
				xlog.Debug("failed to cache measured VRAM", "model", modelName, "error", err)
			}
			return
		}
	}
	xlog.Debug("measured VRAM not cleanly attributable; keeping heuristic", "model", modelName, "node", nodeID)
}
```

Thread `preInstallFree map[int]uint64` and `GPUSet []int` through `scheduleNewModel`→`scheduleAndLoad` (capture pre-install free from the `NodeGPUs` read in Task 3.4; add `GPUSet` to `scheduleLoadResult`).

- [ ] **Step 4: Run test + build**

Run: `go test ./core/services/nodes/ -run TestNodes -v` → PASS
Run: `go build ./...` → success.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/router.go core/services/nodes/router_test.go
git commit -m "feat(nodes): measure and cache per-GPU VRAM footprint after load"
```

---

### Task 4.2: Bump estimate on load failure (escape repeated under-estimate)

**Files:**
- Modify: `core/services/nodes/router.go` (`scheduleAndLoad` LoadModel error path ~196)
- Test: `core/services/nodes/router_test.go`

- [ ] **Step 1: Write the failing test**

```go
	It("bumps the cached estimate when a load fails so the next try reserves more", func() {
		r := newRouterWithFakes()
		f := r.registry.(*fakeModelRouter)
		f.node = &BackendNode{ID: "n", Name: "noctis", Status: StatusHealthy, NodeType: NodeTypeBackend}
		f.nodeGPUs = []NodeGPU{{NodeID: "n", GPUIndex: 0, FreeVRAM: 12 << 30}}
		f.vramEstimate = &ModelVRAMEstimate{ModelName: "m", VRAMBytes: 6 << 30, Source: "heuristic"}
		f.loadErr = fmt.Errorf("CUDA out of memory")

		_, err := r.scheduleAndLoad(context.Background(), "diffusers", "m", "m", &pb.ModelOptions{Model: "m"}, false, 0)
		Expect(err).To(HaveOccurred())
		Expect(f.lastUpsert.VRAMBytes).To(BeNumerically(">", uint64(6<<30)))
	})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: FAIL — no bump on failure.

- [ ] **Step 3: Implement the bump**

In `scheduleAndLoad`, in the `LoadModel` error and `!res.Success` branches, before returning:

```go
		// Under-estimate likely: bump the cached estimate (×1.5, capped) so the
		// next placement reserves/spans more instead of retrying into the same
		// wall. Best-effort.
		if estimatedVRAM > 0 {
			bumped := estimatedVRAM * 3 / 2
			_ = r.registry.UpsertModelVRAMEstimate(ctx, trackingKey, backendType, bumped, "heuristic", 1)
		}
		// release per-GPU reservations (already handled in scheduleNewModel for
		// install errors; for LoadModel errors release here)
```

Note: `estimatedVRAM` must be in scope in `scheduleAndLoad`; if it currently lives only in `scheduleNewModel`, return it (or the `GPUSet` + estimate) via `scheduleLoadResult`/an out param so the failure handler can bump.

- [ ] **Step 4: Run test + build**

Run: `go test ./core/services/nodes/ -run TestNodes -v` → PASS
Run: `go build ./...` → success.

- [ ] **Step 5: Commit**

```bash
git add core/services/nodes/router.go core/services/nodes/router_test.go
git commit -m "feat(nodes): bump cached VRAM estimate on load failure"
```

---

### Task 4.3: Regression test for the incident

**Files:**
- Test: `core/services/nodes/router_test.go`

- [ ] **Step 1: Write the regression test**

```go
	Describe("incident regression: one full GPU + one free GPU", func() {
		It("does not OOM-loop; places the new model on the free GPU", func() {
			r := newRouterWithFakes()
			f := r.registry.(*fakeModelRouter)
			f.node = &BackendNode{ID: "noctis", Name: "noctis-2gpu", Status: StatusHealthy, NodeType: NodeTypeBackend}
			f.nodeGPUs = []NodeGPU{
				{NodeID: "noctis", GPUIndex: 0, FreeVRAM: 156 << 20},  // GPU0 full (the loaded SDXL)
				{NodeID: "noctis", GPUIndex: 1, FreeVRAM: 11900 << 20}, // GPU1 free
			}
			f.vramEstimate = &ModelVRAMEstimate{ModelName: "real-cosplayer", VRAMBytes: 7 << 30, Source: "heuristic"}

			_, _, _, gpuSet, err := r.scheduleNewModel(context.Background(), "diffusers", "real-cosplayer", &pb.ModelOptions{Model: "real-cosplayer"})
			Expect(err).ToNot(HaveOccurred())
			Expect(gpuSet).To(Equal([]int{1}))
		})
	})
```

- [ ] **Step 2: Run it**

Run: `go test ./core/services/nodes/ -run TestNodes -v`
Expected: PASS (validates Phase 3 behavior end-to-end at the unit level).

- [ ] **Step 3: Commit**

```bash
git add core/services/nodes/router_test.go
git commit -m "test(nodes): regression for one-full-one-free-GPU placement"
```

**Phase 4 gate / final:** `go build ./...` and `go test ./...` (or at least `./core/... ./pkg/...`) green. `gofmt -l` reports no files.

---

## Post-implementation manual verification (on the cluster)
1. Roll out the new master + workers.
2. Confirm workers send `gpus[]`: `kubectl logs <master>` shows heartbeats; query `node_gpus` has one row per physical GPU per node.
3. Load two SDXL models onto `noctis-2gpu`; confirm via `nvidia-smi` on the worker that the second lands on GPU1 (both GPUs used), not an OOM loop on GPU0.
4. Load a model larger than 12 GB; confirm it spans both GPUs (`device_map="auto"`, `TensorParallelSize=2`) and loads.
5. Confirm `model_vram_estimates` gains `measured` rows after loads.

## Notes / risks
- **`estimate == 0` must never hard-fail a load.** This was the most dangerous failure mode of an earlier draft: the old code placed unsized models VRAM-blind; the per-GPU path must preserve that liveness. Tasks 3.1/3.4 (largestFreeGPU) and 3.5 (legacy fallback) implement this. The terminal "more VRAM than any GPU set" error fires ONLY when `estimatedVRAM > 0`. Do not "simplify" this away.
- **Task 4.1 measurement concurrency guard is unverified.** The "clean attribution" check (no other concurrent load on the assigned GPUs in the window) is described but the test (`nodeGPUsSeq`) only exercises the happy path. Measurement is best-effort and self-correcting (a noisy reading just gets overwritten by a later clean one, and a wrong-low measured value is caught by the Task 4.2 bump-on-failure), so this is acceptable — but treat the guard as not-yet-proven and add a concurrency test if measured estimates look wrong in practice.
- **Per-GPU reserve-failure proceeds unreserved (double-book risk).** Task 3.4(d) logs and proceeds when `ReserveVRAMOnGPU` fails, inherited from the node-level pattern. Per-GPU this is slightly riskier: two concurrent loads can pick the same GPU and the loser proceeds without a reservation → possible double-book → OOM. Low frequency (tight race window, worker heartbeat reconciles). Consider re-planning (pick a different GPU/node) on reserve-failure instead of proceeding — non-blocking, revisit if OOMs recur under concurrent load.
- `estimateModelVRAM` keys the cache on `opts.Model`; verify this equals the `trackingKey`/model id used at load time (they should match — confirm during Task 2.3).
- `settleDelay` in Task 4.1 should be set from the actual heartbeat interval (search worker config for the interval and align).
- Existing `FindNodeWithVRAM`/`FindNodeWithVRAMFromSet` remain for any non-per-GPU callers and for the mixed-version fallback; the per-GPU path supersedes them in `scheduleNewModel`. Leave them unless a later cleanup removes dead callers.
- **Registry interface additions:** the `SmartRouter` reads the registry through an interface (the one declaring `ReserveVRAM`, `FindNodeWithVRAM`, `SetNodeModel`, etc. — `core/services/nodes/interfaces.go` / `router.go`). Every new method called on `r.registry` must be added there AND to the test fake: `NodeGPUs`, `CandidateNodesByFreeVRAM`, `ReserveVRAMOnGPU`, `ReleaseVRAMOnGPU`, `GetModelVRAMEstimate`, `UpsertModelVRAMEstimate`. Do this in the first task that introduces each call, or the package won't compile.
- **Test harness naming:** `newRouterWithFakes()` / `fakeModelRouter` field names in the test snippets are illustrative — match the existing constructor and fake in `router_test.go` (e.g. the existing `fakeModelRouter` with its `ReserveVRAM`/`SetNodeModel` stubs). Add the new fake fields (`nodeGPUs`, `nodeGPUsSeq`, `vramEstimate`, `lastInstallGPUIndices`, `lastUpsert`, `evictErr`, `loadErr`) alongside the existing ones and have the fake's new methods read/record them.
- **`scheduleNewModel` return arity changed** to `(*BackendNode, string, int, []int, error)` (adds `gpuSet`). Update the sole caller `scheduleAndLoad` and every test that calls it — all 4-value call sites become 5-value.
