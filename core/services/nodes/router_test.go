package nodes

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/mudler/LocalAI/core/services/messaging"
	"github.com/mudler/LocalAI/core/services/nodes/prefixcache"
	"github.com/mudler/LocalAI/core/services/testutil"
	"github.com/mudler/LocalAI/pkg/distributedhdr"
	grpc "github.com/mudler/LocalAI/pkg/grpc"
	pb "github.com/mudler/LocalAI/pkg/grpc/proto"
	ggrpc "google.golang.org/grpc"
	"gorm.io/gorm"
)

// ---------------------------------------------------------------------------
// Fake FileStager (pre-existing)
// ---------------------------------------------------------------------------

// fakeFileStager is a minimal FileStager that records calls and returns
// predictable remote paths without touching the filesystem or network.
type fakeFileStager struct {
	ensureCalls []ensureCall
}

type ensureCall struct {
	nodeID, localPath, key string
}

func (f *fakeFileStager) EnsureRemote(_ context.Context, nodeID, localPath, key string) (string, error) {
	f.ensureCalls = append(f.ensureCalls, ensureCall{nodeID, localPath, key})
	return "/remote/" + key, nil
}

func (f *fakeFileStager) FetchRemote(_ context.Context, _, _, _ string) error { return nil }

func (f *fakeFileStager) FetchRemoteByKey(_ context.Context, _, _, _ string) error { return nil }

func (f *fakeFileStager) AllocRemoteTemp(_ context.Context, _ string) (string, error) {
	return "/remote/tmp", nil
}

func (f *fakeFileStager) StageRemoteToStore(_ context.Context, _, _, _ string) error { return nil }

func (f *fakeFileStager) ListRemoteDir(_ context.Context, _, _ string) ([]string, error) {
	return nil, nil
}

// ---------------------------------------------------------------------------
// Fake ModelRouter
// ---------------------------------------------------------------------------

// fakeModelRouter implements ModelRouter with configurable return values.
type fakeModelRouter struct {
	// mu guards all mutable fields so measurement goroutines don't race test assertions.
	mu sync.Mutex

	// FindAndLockNodeWithModel returns
	findAndLockNode *BackendNode
	findAndLockNM   *NodeModel
	findAndLockErr  error

	// FindNodeWithVRAM returns
	findVRAMNode *BackendNode
	findVRAMErr  error

	// FindIdleNode returns
	findIdleNode *BackendNode
	findIdleErr  error

	// FindLeastLoadedNode returns
	findLeastLoadedNode *BackendNode
	findLeastLoadedErr  error

	// FindGlobalLRUModelWithZeroInFlight returns
	findGlobalLRUModel *NodeModel
	findGlobalLRUErr   error

	// FindLRUModel returns
	findLRUModel *NodeModel
	findLRUErr   error

	// Get returns
	getNode *BackendNode
	getErr  error

	// GetModelScheduling returns
	getModelScheduling *ModelSchedulingConfig
	getModelSchedErr   error

	// FindNodesBySelector returns
	findBySelectorNodes []BackendNode
	findBySelectorErr   error

	// *FromSet variants
	findVRAMFromSetNode        *BackendNode
	findVRAMFromSetErr         error
	findIdleFromSetNode        *BackendNode
	findIdleFromSetErr         error
	findLeastLoadedFromSetNode *BackendNode
	findLeastLoadedFromSetErr  error

	// GetNodeLabels returns
	getNodeLabels    []NodeLabel
	getNodeLabelsErr error

	// FindNodesWithModel returns (keyed by model name)
	findNodesWithModelByName map[string][]BackendNode
	findNodesWithModelErr    error

	// LoadedReplicaStats returns (keyed by model name)
	loadedReplicaStatsByName map[string][]ReplicaCandidate
	loadedReplicaStatsErr    error

	// Track calls for assertions
	decrementCalls []string // "nodeID:modelName"
	incrementCalls []string
	removeCalls    []string
	setCalls       []string
	touchCalls     []string

	// Preferences passed to FindAndLockNodeWithModel, in call order. nil
	// entries are recorded too, so tests can assert "preference was nil".
	findAndLockPrefs []*RoutePreference

	// GetModelVRAMEstimate returns this value when non-nil; gorm.ErrRecordNotFound otherwise.
	vramEstimate *ModelVRAMEstimate

	// Per-GPU placement fakes (Task 3.4).
	// candidateNodes is returned by CandidateNodesByFreeVRAM (already ordered).
	candidateNodes []BackendNode
	candidateErr   error
	// nodeGPUs is returned by NodeGPUs for every node id (the mock-based tests
	// model a single node, so one slice suffices).
	nodeGPUs    []NodeGPU
	nodeGPUsErr error

	// nodeGPUsSeq is an optional sequence of snapshots for NodeGPUs. When non-empty,
	// each call to NodeGPUs pops and returns the first entry; the last entry repeats.
	// Falls back to nodeGPUs when empty.
	nodeGPUsSeq [][]NodeGPU

	// lastUpsert captures the most recent UpsertModelVRAMEstimate call with
	// source=="measured". Protected by mu.
	lastUpsert *ModelVRAMEstimate

	// lastHeuristicUpsert captures the most recent UpsertModelVRAMEstimate call
	// with source=="heuristic". Protected by mu.
	lastHeuristicUpsert *ModelVRAMEstimate
}

func (f *fakeModelRouter) FindAndLockNodeWithModel(_ context.Context, modelName string, _ []string, pref *RoutePreference) (*BackendNode, *NodeModel, error) {
	f.findAndLockPrefs = append(f.findAndLockPrefs, pref)
	return f.findAndLockNode, f.findAndLockNM, f.findAndLockErr
}

func (f *fakeModelRouter) LoadedReplicaStats(_ context.Context, modelName string, _ []string) ([]ReplicaCandidate, error) {
	if f.loadedReplicaStatsErr != nil {
		return nil, f.loadedReplicaStatsErr
	}
	return f.loadedReplicaStatsByName[modelName], nil
}

func (f *fakeModelRouter) DecrementInFlight(_ context.Context, nodeID, modelName string, _ int) error {
	f.decrementCalls = append(f.decrementCalls, nodeID+":"+modelName)
	return nil
}

func (f *fakeModelRouter) IncrementInFlight(_ context.Context, nodeID, modelName string, _ int) error {
	f.incrementCalls = append(f.incrementCalls, nodeID+":"+modelName)
	return nil
}

func (f *fakeModelRouter) RemoveNodeModel(_ context.Context, nodeID, modelName string, _ int) error {
	f.removeCalls = append(f.removeCalls, nodeID+":"+modelName)
	return nil
}

func (f *fakeModelRouter) RemoveAllNodeModelReplicas(_ context.Context, nodeID, modelName string) error {
	// Same recorded key as RemoveNodeModel so existing tests that assert "the
	// model was removed" don't need to know whether the production code used
	// the per-replica or all-replicas variant.
	f.removeCalls = append(f.removeCalls, nodeID+":"+modelName)
	return nil
}

func (f *fakeModelRouter) TouchNodeModel(_ context.Context, nodeID, modelName string, _ int) {
	f.touchCalls = append(f.touchCalls, nodeID+":"+modelName)
}

func (f *fakeModelRouter) SetNodeModel(_ context.Context, nodeID, modelName string, _ int, state, address string, _ int, _ string) error {
	f.setCalls = append(f.setCalls, fmt.Sprintf("%s:%s:%s:%s", nodeID, modelName, state, address))
	return nil
}

func (f *fakeModelRouter) SetNodeModelLoadInfo(_ context.Context, _, _ string, _ int, _ string, _ []byte) error {
	return nil
}

func (f *fakeModelRouter) UpsertModelLoadInfo(_ context.Context, _, _ string, _ []byte) error {
	return nil
}

func (f *fakeModelRouter) GetModelLoadInfo(_ context.Context, _ string) (string, []byte, error) {
	return "", nil, fmt.Errorf("not found")
}

func (f *fakeModelRouter) NextFreeReplicaIndex(_ context.Context, _, _ string, _ int) (int, error) {
	return 0, nil
}

func (f *fakeModelRouter) CountReplicasOnNode(_ context.Context, _, _ string) (int, error) {
	return 0, nil
}

func (f *fakeModelRouter) FindNodeWithVRAM(_ context.Context, _ uint64) (*BackendNode, error) {
	return f.findVRAMNode, f.findVRAMErr
}

func (f *fakeModelRouter) FindIdleNode(_ context.Context) (*BackendNode, error) {
	return f.findIdleNode, f.findIdleErr
}

func (f *fakeModelRouter) FindLeastLoadedNode(_ context.Context) (*BackendNode, error) {
	return f.findLeastLoadedNode, f.findLeastLoadedErr
}

func (f *fakeModelRouter) FindGlobalLRUModelWithZeroInFlight(_ context.Context) (*NodeModel, error) {
	return f.findGlobalLRUModel, f.findGlobalLRUErr
}

func (f *fakeModelRouter) FindLRUModel(_ context.Context, _ string) (*NodeModel, error) {
	return f.findLRUModel, f.findLRUErr
}

func (f *fakeModelRouter) Get(_ context.Context, _ string) (*BackendNode, error) {
	return f.getNode, f.getErr
}

func (f *fakeModelRouter) GetModelScheduling(_ context.Context, _ string) (*ModelSchedulingConfig, error) {
	return f.getModelScheduling, f.getModelSchedErr
}

func (f *fakeModelRouter) FindNodesBySelector(_ context.Context, _ map[string]string) ([]BackendNode, error) {
	return f.findBySelectorNodes, f.findBySelectorErr
}

func (f *fakeModelRouter) FindNodesWithFreeSlot(_ context.Context, _ string, _ []string) ([]BackendNode, error) {
	// Default: same answer as FindNodesBySelector. Tests that need a
	// specific filter can override by reusing findBySelectorNodes.
	return f.findBySelectorNodes, f.findBySelectorErr
}

func (f *fakeModelRouter) ReserveVRAM(_ context.Context, _ string, _ uint64) error {
	return nil
}

func (f *fakeModelRouter) ReleaseVRAM(_ context.Context, _ string, _ uint64) error {
	return nil
}

func (f *fakeModelRouter) FindNodeWithVRAMFromSet(_ context.Context, _ uint64, _ []string) (*BackendNode, error) {
	return f.findVRAMFromSetNode, f.findVRAMFromSetErr
}

func (f *fakeModelRouter) FindIdleNodeFromSet(_ context.Context, _ []string) (*BackendNode, error) {
	return f.findIdleFromSetNode, f.findIdleFromSetErr
}

func (f *fakeModelRouter) FindLeastLoadedNodeFromSet(_ context.Context, _ []string) (*BackendNode, error) {
	return f.findLeastLoadedFromSetNode, f.findLeastLoadedFromSetErr
}

func (f *fakeModelRouter) GetNodeLabels(_ context.Context, _ string) ([]NodeLabel, error) {
	return f.getNodeLabels, f.getNodeLabelsErr
}

func (f *fakeModelRouter) FindNodesWithModel(_ context.Context, modelName string) ([]BackendNode, error) {
	if f.findNodesWithModelErr != nil {
		return nil, f.findNodesWithModelErr
	}
	return f.findNodesWithModelByName[modelName], nil
}

func (f *fakeModelRouter) GetModelVRAMEstimate(_ context.Context, _ string) (*ModelVRAMEstimate, error) {
	if f.vramEstimate == nil {
		return nil, gorm.ErrRecordNotFound
	}
	return f.vramEstimate, nil
}

func (f *fakeModelRouter) UpsertModelVRAMEstimate(_ context.Context, modelName, backend string, bytes uint64, source string, gpuCount int) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	est := &ModelVRAMEstimate{
		ModelName:        modelName,
		Backend:          backend,
		VRAMBytes:        bytes,
		Source:           source,
		GPUCountObserved: gpuCount,
	}
	if source == "measured" {
		f.lastUpsert = est
	}
	if source == "heuristic" {
		f.lastHeuristicUpsert = est
	}
	return nil
}

func (f *fakeModelRouter) CandidateNodesByFreeVRAM(_ context.Context, _ []string) ([]BackendNode, error) {
	return f.candidateNodes, f.candidateErr
}

func (f *fakeModelRouter) NodeGPUs(_ context.Context, _ string) ([]NodeGPU, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.nodeGPUsSeq) > 0 {
		snap := f.nodeGPUsSeq[0]
		if len(f.nodeGPUsSeq) > 1 {
			f.nodeGPUsSeq = f.nodeGPUsSeq[1:]
		}
		return snap, f.nodeGPUsErr
	}
	return f.nodeGPUs, f.nodeGPUsErr
}

func (f *fakeModelRouter) ReserveVRAMOnGPU(_ context.Context, _ string, _ int, _ uint64) error {
	return nil
}

func (f *fakeModelRouter) ReleaseVRAMOnGPU(_ context.Context, _ string, _ int, _ uint64) error {
	return nil
}

// fakeConflictResolver implements ConcurrencyConflictResolver from a static map.
type fakeConflictResolver struct {
	conflicts map[string][]string
}

func (f *fakeConflictResolver) GetModelsConflictingWith(name string) []string {
	if f == nil {
		return nil
	}
	return f.conflicts[name]
}

// ---------------------------------------------------------------------------
// Fake BackendClientFactory + Backend
// ---------------------------------------------------------------------------

// stubBackend implements grpc.Backend with configurable HealthCheck and LoadModel.
type stubBackend struct {
	grpc.Backend // embed to satisfy interface; unused methods will panic if called

	healthResult bool
	healthErr    error
	loadResult   *pb.Result
	loadErr      error
}

func (f *stubBackend) HealthCheck(_ context.Context) (bool, error) {
	return f.healthResult, f.healthErr
}

func (f *stubBackend) LoadModel(_ context.Context, _ *pb.ModelOptions, _ ...ggrpc.CallOption) (*pb.Result, error) {
	return f.loadResult, f.loadErr
}

func (f *stubBackend) IsBusy() bool { return false }

// stubClientFactory returns the same stubBackend for every call.
type stubClientFactory struct {
	client *stubBackend
}

func (f *stubClientFactory) NewClient(_ string, _ bool) grpc.Backend {
	return f.client
}

// ---------------------------------------------------------------------------
// Fake NodeCommandSender (unloader)
// ---------------------------------------------------------------------------

type fakeUnloader struct {
	// mu guards installCalls and upgradeCalls so concurrent test
	// goroutines (e.g. singleflight specs) don't race the slice appends.
	mu sync.Mutex

	installReply *messaging.BackendInstallReply
	installErr   error
	installCalls []installCall // every InstallBackend invocation, in order
	// installHook, if non-nil, runs at the start of InstallBackend before
	// the call is recorded. Used by concurrency tests as a deterministic
	// "block here" seam — set installHook to a function that sleeps or
	// blocks on a channel to overlap two callers.
	installHook func()

	upgradeReply *messaging.BackendUpgradeReply
	upgradeErr   error
	upgradeCalls []upgradeCall // every UpgradeBackend invocation, in order

	stopCalls   []string // "nodeID:model"
	stopErr     error
	unloadCalls []string
	unloadErr   error

	// lastInstallGPUIndices captures the GPU indices the scheduler pinned on
	// the most recent InstallBackend call (Task 3.4).
	lastInstallGPUIndices []int
}

// installCall captures the args we care about when asserting that the
// reconciler / router did or did not fire a NATS install. The fake records
// every call so tests can verify both presence and shape (e.g. that backend
// is non-empty).
type installCall struct {
	nodeID  string
	backend string
	modelID string
	replica int
}

type upgradeCall struct {
	nodeID  string
	backend string
	replica int
}

func (f *fakeUnloader) InstallBackend(nodeID, backend, modelID, _, _, _, _ string, replica int, gpuIndices []int, _ string, _ func(messaging.BackendInstallProgressEvent)) (*messaging.BackendInstallReply, error) {
	// installHook intentionally runs OUTSIDE the mutex: the hook may block
	// on a channel and we don't want to serialize concurrent callers,
	// which would defeat the singleflight-overlap test.
	if f.installHook != nil {
		f.installHook()
	}
	f.mu.Lock()
	f.installCalls = append(f.installCalls, installCall{nodeID, backend, modelID, replica})
	f.lastInstallGPUIndices = gpuIndices
	f.mu.Unlock()
	return f.installReply, f.installErr
}

func (f *fakeUnloader) UpgradeBackend(nodeID, backend, _, _, _, _ string, replica int) (*messaging.BackendUpgradeReply, error) {
	f.mu.Lock()
	f.upgradeCalls = append(f.upgradeCalls, upgradeCall{nodeID, backend, replica})
	f.mu.Unlock()
	return f.upgradeReply, f.upgradeErr
}

func (f *fakeUnloader) DeleteBackend(_, _ string) (*messaging.BackendDeleteReply, error) {
	return &messaging.BackendDeleteReply{Success: true}, nil
}

func (f *fakeUnloader) ListBackends(_ string) (*messaging.BackendListReply, error) {
	return &messaging.BackendListReply{}, nil
}

func (f *fakeUnloader) StopBackend(nodeID, backend string) error {
	f.stopCalls = append(f.stopCalls, nodeID+":"+backend)
	return f.stopErr
}

func (f *fakeUnloader) UnloadModelOnNode(nodeID, modelName string) error {
	f.unloadCalls = append(f.unloadCalls, nodeID+":"+modelName)
	return f.unloadErr
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

var _ = Describe("SmartRouter", func() {
	// -----------------------------------------------------------------------
	// Unit tests using mock interfaces (no DB required)
	// -----------------------------------------------------------------------
	Describe("Route (mock-based)", func() {
		var (
			reg      *fakeModelRouter
			backend  *stubBackend
			factory  *stubClientFactory
			unloader *fakeUnloader
		)

		BeforeEach(func() {
			reg = &fakeModelRouter{}
			backend = &stubBackend{}
			factory = &stubClientFactory{client: backend}
			unloader = &fakeUnloader{
				installReply: &messaging.BackendInstallReply{
					Success: true,
					Address: "10.0.0.1:9001",
				},
			}
		})

		Context("model already loaded on a healthy node", func() {
			It("returns the client and a release function", func() {
				node := &BackendNode{ID: "n1", Name: "node-1", Address: "10.0.0.1:50051"}
				nm := &NodeModel{NodeID: "n1", ModelName: "my-model", Address: "10.0.0.1:9001"}
				reg.findAndLockNode = node
				reg.findAndLockNM = nm
				backend.healthResult = true

				router := NewSmartRouter(reg, SmartRouterOptions{
					Unloader:      unloader,
					ClientFactory: factory,
				})

				result, err := router.Route(context.Background(), "my-model", "models/my-model.gguf", "llama-cpp", nil, false)
				Expect(err).ToNot(HaveOccurred())
				Expect(result).ToNot(BeNil())
				Expect(result.Node.ID).To(Equal("n1"))

				// TouchNodeModel should have been called
				Expect(reg.touchCalls).To(ContainElement("n1:my-model"))

				// The initial in-flight reservation from FindAndLockNodeWithModel is released
				// after the first inference call completes via OnFirstComplete callback.
				// Release only closes the client.
				result.Release()
				// No decrement on Release — it happens via OnFirstComplete after first Predict
				Expect(reg.decrementCalls).To(BeEmpty())
			})
		})

		Context("model not loaded, falls through to scheduling", func() {
			It("schedules on an idle node and records the model", func() {
				// FindAndLockNodeWithModel always fails — simulates no cached model
				// (equivalent to the health-check-failure fallthrough path).
				idleNode := &BackendNode{ID: "n2", Name: "idle-node", Address: "10.0.0.2:50051"}
				reg2 := &fakeModelRouter{
					findAndLockErr: errors.New("not found"),
					findIdleNode:   idleNode,
				}
				backend.loadResult = &pb.Result{Success: true}

				router := NewSmartRouter(reg2, SmartRouterOptions{
					Unloader:      unloader,
					ClientFactory: factory,
				})

				result, err := router.Route(context.Background(), "some-model", "models/some-model.gguf", "llama-cpp", nil, false)
				Expect(err).ToNot(HaveOccurred())
				Expect(result).ToNot(BeNil())
				Expect(result.Node.ID).To(Equal("n2"))

				// SetNodeModel should record the model as loaded on the node
				Expect(reg2.setCalls).To(HaveLen(1))
				Expect(reg2.setCalls[0]).To(ContainSubstring("n2:some-model:loaded"))
			})
		})

		Context("model not loaded, no DB (advisory lock bypassed)", func() {
			It("schedules on an available node via FindIdleNode", func() {
				reg.findAndLockErr = errors.New("not found")
				idleNode := &BackendNode{ID: "n3", Name: "idle", Address: "10.0.0.3:50051"}
				reg.findIdleNode = idleNode
				backend.loadResult = &pb.Result{Success: true}

				router := NewSmartRouter(reg, SmartRouterOptions{
					Unloader:      unloader,
					ClientFactory: factory,
					// DB is nil — no advisory lock
				})

				result, err := router.Route(context.Background(), "new-model", "models/new.gguf", "llama-cpp", nil, false)
				Expect(err).ToNot(HaveOccurred())
				Expect(result.Node.ID).To(Equal("n3"))
			})
		})
	})

	Describe("scheduleNewModel (mock-based, via Route)", func() {
		var (
			reg      *fakeModelRouter
			backend  *stubBackend
			factory  *stubClientFactory
			unloader *fakeUnloader
		)

		BeforeEach(func() {
			reg = &fakeModelRouter{
				findAndLockErr: errors.New("not found"),
			}
			backend = &stubBackend{
				loadResult: &pb.Result{Success: true},
			}
			factory = &stubClientFactory{client: backend}
			unloader = &fakeUnloader{
				installReply: &messaging.BackendInstallReply{
					Success: true,
					Address: "10.0.0.1:9001",
				},
			}
		})

		It("finds a node with sufficient VRAM first", func() {
			vramNode := &BackendNode{ID: "vram-node", Name: "gpu-box", Address: "10.0.0.10:50051"}
			reg.findVRAMNode = vramNode

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
			})

			// Pass non-nil ModelOptions so estimateModelVRAM runs (returns 0 for
			// missing files, so FindNodeWithVRAM won't actually be called unless
			// estimatedVRAM > 0). To trigger VRAM path we need estimatedVRAM > 0,
			// but that requires real files. Instead test the fallback: VRAM returns
			// error, idle succeeds.
			// Actually, estimateModelVRAM returns 0 when model files don't exist,
			// so the VRAM branch is skipped and we go to idle/least-loaded.
			// To properly test VRAM path, we'd need to mock estimateModelVRAM.
			// For now, verify the fallback paths work correctly.

			// With no real model files, estimatedVRAM=0, so VRAM path is skipped.
			// Set idle node to test that path.
			reg.findVRAMNode = nil
			reg.findVRAMErr = errors.New("no vram nodes")
			idleNode := &BackendNode{ID: "idle-vram", Name: "idle", Address: "10.0.0.11:50051"}
			reg.findIdleNode = idleNode

			result, err := router.Route(context.Background(), "m1", "models/m1.gguf", "llama-cpp", &pb.ModelOptions{}, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Node.ID).To(Equal("idle-vram"))
		})

		It("falls back to idle when VRAM search fails", func() {
			reg.findVRAMErr = errors.New("no vram")
			idleNode := &BackendNode{ID: "idle-1", Name: "idle-node", Address: "10.0.0.20:50051"}
			reg.findIdleNode = idleNode

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
			})

			result, err := router.Route(context.Background(), "m2", "models/m2.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Node.ID).To(Equal("idle-1"))
		})

		It("falls back to least-loaded when both VRAM and idle fail", func() {
			reg.findVRAMErr = errors.New("no vram")
			reg.findIdleErr = errors.New("no idle")
			llNode := &BackendNode{ID: "ll-1", Name: "least-loaded", Address: "10.0.0.30:50051"}
			reg.findLeastLoadedNode = llNode

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
			})

			result, err := router.Route(context.Background(), "m3", "models/m3.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Node.ID).To(Equal("ll-1"))
		})

		It("returns error when no nodes are available and no DB for eviction", func() {
			reg.findVRAMErr = errors.New("no vram")
			reg.findIdleErr = errors.New("no idle")
			reg.findLeastLoadedErr = errors.New("no nodes")

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
				// DB is nil — evictLRUAndFreeNode will fail because r.db is nil
			})

			_, err := router.Route(context.Background(), "m4", "models/m4.gguf", "llama-cpp", nil, false)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no available nodes"))
		})

		It("with estimatedVRAM>0 and no node fitting, skips idle/least-loaded fallback (goes to eviction)", func() {
			// VRAM filter returns no node; idle and least-loaded would succeed
			// if consulted, but the patched scheduler must skip them because we
			// have a VRAM estimate that says they're too small to fit the model.
			// With DB=nil, eviction returns ErrEvictionBusy and the call fails
			// rather than picking a known-too-small node and OOMing at load.
			reg.findVRAMErr = errors.New("no node fits")
			reg.findIdleNode = &BackendNode{ID: "small-but-idle", Name: "idle"}
			reg.findLeastLoadedNode = &BackendNode{ID: "small-but-ll", Name: "ll"}

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
				VRAMEstimator: func(_ context.Context, _ *pb.ModelOptions) uint64 {
					return 25 * 1024 * 1024 * 1024 // 25 GiB — more than any test node has
				},
				// DB nil → eviction returns ErrEvictionBusy, so Route fails.
			})

			_, err := router.Route(context.Background(), "m-vram", "models/m-vram.gguf",
				"llama-cpp", &pb.ModelOptions{}, false)
			Expect(err).To(HaveOccurred())
			// Must NOT have picked the known-too-small idle / least-loaded nodes.
			Expect(err.Error()).ToNot(ContainSubstring("small-but-idle"))
			Expect(err.Error()).ToNot(ContainSubstring("small-but-ll"))
		})
	})

	Describe("per-GPU placement (mock-based)", func() {
		const GiB = uint64(1) << 30

		It("places a single-GPU model on the free GPU, not the full one", func() {
			node := &BackendNode{ID: "n", Name: "noctis", Status: StatusHealthy, NodeType: NodeTypeBackend}
			reg := &fakeModelRouter{
				findAndLockErr: errors.New("not found"),
				candidateNodes: []BackendNode{*node},
				nodeGPUs: []NodeGPU{
					{NodeID: "n", GPUIndex: 0, FreeVRAM: 200 << 20}, // GPU0 full
					{NodeID: "n", GPUIndex: 1, FreeVRAM: 11 * GiB},  // GPU1 free
				},
				vramEstimate: &ModelVRAMEstimate{ModelName: "m", VRAMBytes: 7 * GiB, Source: "measured"},
			}
			unloader := &fakeUnloader{
				installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:9001"},
			}
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: &stubClientFactory{client: &stubBackend{loadResult: &pb.Result{Success: true}}},
			})

			gotNode, _, _, gpuSet, err := router.scheduleNewModel(context.Background(), "diffusers", "m", &pb.ModelOptions{Model: "m"})
			Expect(err).ToNot(HaveOccurred())
			Expect(gotNode.ID).To(Equal("n"))
			Expect(gpuSet).To(Equal([]int{1}))
			// The pinned index must reach the worker via backend.install.
			Expect(unloader.lastInstallGPUIndices).To(Equal([]int{1}))
		})

		// Task 3.5: terminal error when a real estimate exceeds every GPU set.
		It("returns a clear terminal error when no GPU set on any node fits the estimate", func() {
			// One node with 2 GiB free, model needs 40 GiB.
			// sawPerGPUNode=true (node has GPU data), planGPUSet fails → eviction.
			// DB=nil → evictLRUAndFreeNode returns ErrEvictionBusy every attempt.
			// After the eviction loop, node==nil AND estimatedVRAM>0 → terminal error.
			node := &BackendNode{ID: "n", Name: "small", Status: StatusHealthy, NodeType: NodeTypeBackend}
			reg := &fakeModelRouter{
				findAndLockErr: errors.New("not found"),
				candidateNodes: []BackendNode{*node},
				nodeGPUs:       []NodeGPU{{NodeID: "n", GPUIndex: 0, FreeVRAM: 2 * GiB}},
				vramEstimate:   &ModelVRAMEstimate{ModelName: "big", VRAMBytes: 40 * GiB, Source: "measured"},
			}
			unloader := &fakeUnloader{
				installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:9001"},
			}
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: &stubClientFactory{client: &stubBackend{loadResult: &pb.Result{Success: true}}},
				// DB intentionally nil: evictLRUAndFreeNode returns ErrEvictionBusy.
			})

			_, _, _, _, err := router.scheduleNewModel(context.Background(), "diffusers", "big", &pb.ModelOptions{Model: "big"})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("more VRAM than any GPU set"))
		})

		// Task 3.5: estimate==0 must never hard-fail — falls back to legacy placement.
		It("falls back to legacy idle-node placement when estimate is zero and per-GPU data fits nothing", func() {
			// Node has one GPU with zero effective-free VRAM → largestFreeGPU fails.
			// vramEstimate=nil + no ModelFile → estimatedVRAM=0.
			// After the per-GPU and mixed-version blocks both skip, node==nil AND
			// estimatedVRAM==0 → legacy fallback to FindIdleNode succeeds.
			legacyNode := &BackendNode{ID: "legacy", Name: "legacy-node", Status: StatusHealthy, NodeType: NodeTypeBackend}
			reg := &fakeModelRouter{
				findAndLockErr: errors.New("not found"),
				candidateNodes: []BackendNode{{ID: "n", Name: "n", Status: StatusHealthy, NodeType: NodeTypeBackend}},
				nodeGPUs:       []NodeGPU{{NodeID: "n", GPUIndex: 0, FreeVRAM: 0}},
				// vramEstimate nil → GetModelVRAMEstimate returns gorm.ErrRecordNotFound → estimatedVRAM=0
				findIdleNode: legacyNode,
			}
			unloader := &fakeUnloader{
				installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:9001"},
			}
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: &stubClientFactory{client: &stubBackend{loadResult: &pb.Result{Success: true}}},
			})

			gotNode, _, _, gpuSet, err := router.scheduleNewModel(context.Background(), "diffusers", "unknown-size", &pb.ModelOptions{Model: "unknown-size"})
			Expect(err).ToNot(HaveOccurred())
			Expect(gotNode.ID).To(Equal("legacy"))
			Expect(gpuSet).To(BeEmpty(), "legacy fallback must not pin a GPU (no estimate available)")
		})

		// Task 4.1: after a successful load, the VRAM footprint is measured and cached.
		It("caches a measured VRAM estimate after a successful load (Task 4.1)", func() {
			// Node has two GPUs: GPU0 is nearly full, GPU1 has 12 GiB free pre-load.
			// The vramEstimate (6 GiB) is used for placement so the scheduler picks GPU1.
			//
			// NodeGPUs sequence:
			//   call 1 (scheduleNewModel placement loop): pre snapshot — GPU1 free=12 GiB
			//   call 2 (scheduleAndLoad pre-load capture): pre snapshot — GPU1 free=12 GiB
			//   call 3+ (recordMeasuredVRAM post-load poll): post snapshot — GPU1 free=4 GiB
			//
			// Expected measured footprint: 12 GiB - 4 GiB = 8 GiB.
			const GiB2 = uint64(1) << 30
			preSnap := []NodeGPU{
				{NodeID: "n", GPUIndex: 0, FreeVRAM: 200 << 20},
				{NodeID: "n", GPUIndex: 1, FreeVRAM: 12 * GiB2},
			}
			postSnap := []NodeGPU{
				{NodeID: "n", GPUIndex: 0, FreeVRAM: 200 << 20},
				{NodeID: "n", GPUIndex: 1, FreeVRAM: 4 * GiB2},
			}

			node := &BackendNode{ID: "n", Name: "noctis", Status: StatusHealthy, NodeType: NodeTypeBackend}
			f := &fakeModelRouter{
				findAndLockErr: errors.New("not found"),
				candidateNodes: []BackendNode{*node},
				// Two pre-snapshots (placement + pre-capture), then post repeats.
				nodeGPUsSeq: [][]NodeGPU{preSnap, preSnap, postSnap},
				// 6 GiB measured estimate drives placement onto GPU1 (12 GiB free).
				vramEstimate: &ModelVRAMEstimate{ModelName: "m", VRAMBytes: 6 * GiB2, Source: "measured"},
			}
			unloader := &fakeUnloader{
				installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:9001"},
			}
			router := NewSmartRouter(f, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: &stubClientFactory{client: &stubBackend{loadResult: &pb.Result{Success: true}}},
			})
			// Speed up the measurement loop so the test finishes quickly.
			router.measureSettleAttempts = 3
			router.measureSettleDelay = time.Millisecond

			result, err := router.scheduleAndLoad(context.Background(), "diffusers", "m", "m",
				&pb.ModelOptions{Model: "m"}, false, 1)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Node.ID).To(Equal("n"))
			Expect(result.GPUSet).To(Equal([]int{1}))

			// The measurement goroutine fires in the background; wait for it to write.
			Eventually(func() *ModelVRAMEstimate {
				f.mu.Lock()
				defer f.mu.Unlock()
				return f.lastUpsert
			}, "2s", "10ms").ShouldNot(BeNil())

			f.mu.Lock()
			upsert := f.lastUpsert
			f.mu.Unlock()
			Expect(upsert.Source).To(Equal("measured"))
			Expect(upsert.VRAMBytes).To(Equal(8*GiB2), "expected 12 GiB - 4 GiB = 8 GiB footprint")
		})

		// Task 4.2: on load failure, bump the heuristic VRAM estimate ×1.5.
		It("bumps the cached heuristic VRAM estimate ×1.5 after a load failure (Task 4.2)", func() {
			// Node has one GPU with 12 GiB free — enough to fit the 6 GiB heuristic
			// estimate, so placement succeeds and we reach LoadModel.
			// LoadModel returns an error, triggering bumpEstimateOnLoadFailure.
			// Expected bump: 6 GiB + 6 GiB/2 = 9 GiB.
			const GiB4 = uint64(1) << 30
			node := &BackendNode{ID: "n", Name: "noctis", Status: StatusHealthy, NodeType: NodeTypeBackend}
			f := &fakeModelRouter{
				findAndLockErr: errors.New("not found"),
				candidateNodes: []BackendNode{*node},
				nodeGPUs: []NodeGPU{
					{NodeID: "n", GPUIndex: 0, FreeVRAM: 12 * GiB4},
				},
				// Heuristic estimate of 6 GiB drives placement.
				vramEstimate: &ModelVRAMEstimate{ModelName: "m", VRAMBytes: 6 * GiB4, Source: "heuristic", GPUCountObserved: 1},
			}
			unloader := &fakeUnloader{
				installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:9001"},
			}
			// LoadModel fails: simulates an OOM or backend rejection.
			failClient := &stubBackend{loadErr: errors.New("worker OOM")}
			router := NewSmartRouter(f, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: &stubClientFactory{client: failClient},
			})

			_, err := router.scheduleAndLoad(context.Background(), "diffusers", "m", "m",
				&pb.ModelOptions{Model: "m"}, false, 1)
			Expect(err).To(HaveOccurred(), "load failure must propagate as an error")

			// The bump is synchronous (called before return), so no Eventually needed.
			f.mu.Lock()
			upsert := f.lastHeuristicUpsert
			f.mu.Unlock()
			Expect(upsert).ToNot(BeNil(), "bumpEstimateOnLoadFailure must have written a heuristic upsert")
			Expect(upsert.VRAMBytes).To(Equal(9*GiB4), "expected 6 GiB * 1.5 = 9 GiB bumped estimate")
		})
	})

	Describe("UnloadModel (mock-based)", func() {
		It("calls StopBackend and removes the model from the registry", func() {
			reg := &fakeModelRouter{}
			unloader := &fakeUnloader{}

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader: unloader,
			})

			err := router.UnloadModel(context.Background(), "node-1", "model-a")
			Expect(err).ToNot(HaveOccurred())

			Expect(unloader.stopCalls).To(ContainElement("node-1:model-a"))
			Expect(reg.removeCalls).To(ContainElement("node-1:model-a"))
		})

		It("returns error when no unloader is configured", func() {
			reg := &fakeModelRouter{}
			router := NewSmartRouter(reg, SmartRouterOptions{})

			err := router.UnloadModel(context.Background(), "node-1", "model-a")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no remote unloader"))
		})
	})

	Describe("EvictLRU (mock-based)", func() {
		It("finds LRU model and unloads it", func() {
			reg := &fakeModelRouter{
				findLRUModel: &NodeModel{NodeID: "n1", ModelName: "old-model"},
			}
			unloader := &fakeUnloader{}

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader: unloader,
			})

			evicted, err := router.EvictLRU(context.Background(), "n1")
			Expect(err).ToNot(HaveOccurred())
			Expect(evicted).To(Equal("old-model"))
			Expect(unloader.stopCalls).To(ContainElement("n1:old-model"))
			Expect(reg.removeCalls).To(ContainElement("n1:old-model"))
		})

		It("returns error when no LRU model is found", func() {
			reg := &fakeModelRouter{
				findLRUErr: errors.New("no models loaded"),
			}

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader: &fakeUnloader{},
			})

			_, err := router.EvictLRU(context.Background(), "n1")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("finding LRU model"))
		})
	})

	Describe("scheduleNewModel with node selector (mock-based, via Route)", func() {
		var (
			reg      *fakeModelRouter
			backend  *stubBackend
			factory  *stubClientFactory
			unloader *fakeUnloader
		)

		BeforeEach(func() {
			reg = &fakeModelRouter{
				findAndLockErr: errors.New("not found"),
			}
			backend = &stubBackend{
				loadResult: &pb.Result{Success: true},
			}
			factory = &stubClientFactory{client: backend}
			unloader = &fakeUnloader{
				installReply: &messaging.BackendInstallReply{
					Success: true,
					Address: "10.0.0.1:9001",
				},
			}
		})

		It("uses *FromSet methods when model has a node selector", func() {
			gpuNode := &BackendNode{ID: "gpu-1", Name: "gpu-node", Address: "10.0.0.50:50051"}
			reg.getModelScheduling = &ModelSchedulingConfig{
				ModelName:    "selector-model",
				NodeSelector: `{"gpu.vendor":"nvidia"}`,
			}
			reg.findBySelectorNodes = []BackendNode{*gpuNode}
			reg.findIdleFromSetNode = gpuNode

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
			})

			result, err := router.Route(context.Background(), "selector-model", "models/selector.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())
			Expect(result.Node.ID).To(Equal("gpu-1"))
		})

		It("returns error when no nodes match selector", func() {
			reg.getModelScheduling = &ModelSchedulingConfig{
				ModelName:    "no-match-model",
				NodeSelector: `{"gpu.vendor":"tpu"}`,
			}
			reg.findBySelectorNodes = nil
			reg.findBySelectorErr = nil

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
			})

			_, err := router.Route(context.Background(), "no-match-model", "models/nomatch.gguf", "llama-cpp", nil, false)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no healthy nodes match selector"))
		})

		It("uses regular methods when model has no scheduling config", func() {
			reg.getModelScheduling = nil
			idleNode := &BackendNode{ID: "regular-1", Name: "regular-node", Address: "10.0.0.60:50051"}
			reg.findIdleNode = idleNode

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
			})

			result, err := router.Route(context.Background(), "regular-model", "models/regular.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())
			Expect(result.Node.ID).To(Equal("regular-1"))
		})
	})

	Describe("Route with selector validation on cached model (mock-based)", func() {
		It("falls through when cached node no longer matches selector", func() {
			cachedNode := &BackendNode{ID: "n-old", Name: "old-node", Address: "10.0.0.70:50051"}
			newNode := &BackendNode{ID: "n-new", Name: "new-node", Address: "10.0.0.71:50051"}

			backend := &stubBackend{
				healthResult: true,
				loadResult:   &pb.Result{Success: true},
			}
			factory := &stubClientFactory{client: backend}
			unloader := &fakeUnloader{
				installReply: &messaging.BackendInstallReply{
					Success: true,
					Address: "10.0.0.71:9001",
				},
			}

			reg := &fakeModelRouter{
				// Step 1: cached model found on old node
				findAndLockNode: cachedNode,
				findAndLockNM:   &NodeModel{NodeID: "n-old", ModelName: "sel-model", Address: "10.0.0.70:9001"},
				// Scheduling config with selector that old node does NOT match
				getModelScheduling: &ModelSchedulingConfig{
					ModelName:    "sel-model",
					NodeSelector: `{"gpu.vendor":"nvidia"}`,
				},
				// Old node has no labels matching the selector
				getNodeLabels: []NodeLabel{
					{NodeID: "n-old", Key: "gpu.vendor", Value: "amd"},
				},
				// For scheduling fallthrough: selector matches new node
				findBySelectorNodes: []BackendNode{*newNode},
				findIdleFromSetNode: newNode,
			}

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: factory,
			})

			result, err := router.Route(context.Background(), "sel-model", "models/sel.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())
			// Should have fallen through to the new node
			Expect(result.Node.ID).To(Equal("n-new"))
			// Old node should have had its in-flight decremented
			Expect(reg.decrementCalls).To(ContainElement("n-old:sel-model"))
		})
	})

	Describe("ScheduleAndLoadModel (mock-based)", func() {
		It("returns an error and does not fire a NATS install when no load info is stored", func() {
			// Reproduces the reconciler scale-up bug: when GetModelLoadInfo
			// returns ErrRecordNotFound (no replica has ever been loaded),
			// the previous fallback called scheduleNewModel with an empty
			// backend type, which the worker rejected on every reconciler
			// tick. The fix bails out cleanly with an explanatory error and
			// never sends backend.install.
			unloader := &fakeUnloader{}
			reg := &fakeModelRouter{}
			router := NewSmartRouter(reg, SmartRouterOptions{Unloader: unloader})

			node, err := router.ScheduleAndLoadModel(context.Background(), "never-loaded", nil)

			Expect(err).To(HaveOccurred())
			Expect(node).To(BeNil())
			Expect(err.Error()).To(ContainSubstring("never-loaded"))
			Expect(unloader.installCalls).To(BeEmpty(),
				"reconciler must not fire backend.install when there is no load info to replicate")
		})
	})

	// -----------------------------------------------------------------------
	// estimateModelVRAM unit tests (mock-based)
	// -----------------------------------------------------------------------
	Describe("estimateModelVRAM", func() {
		It("returns the cached measured estimate when present in the registry", func() {
			reg := &fakeModelRouter{
				vramEstimate: &ModelVRAMEstimate{
					ModelName: "m",
					VRAMBytes: 7_000_000_000,
					Source:    "measured",
				},
			}
			router := NewSmartRouter(reg, SmartRouterOptions{})

			got := router.estimateModelVRAM(context.Background(), &pb.ModelOptions{Model: "m"})
			Expect(got).To(Equal(uint64(7_000_000_000)))
		})
	})

	// -----------------------------------------------------------------------
	// Integration tests using real PostgreSQL (existing)
	// -----------------------------------------------------------------------
	Describe("evictLRUAndFreeNode (integration)", func() {
		var (
			db       *gorm.DB
			registry *NodeRegistry
		)

		BeforeEach(func() {
			if runtime.GOOS == "darwin" {
				Skip("testcontainers requires Docker, not available on macOS CI")
			}
			db = testutil.SetupTestDB()
			var err error
			registry, err = NewNodeRegistry(db)
			Expect(err).ToNot(HaveOccurred())
		})

		It("returns ErrEvictionBusy in under 5 seconds when all models are busy", func() {
			node := &BackendNode{
				Name:     "busy-evict",
				NodeType: NodeTypeBackend,
				Address:  "10.0.0.100:50051",
			}
			Expect(registry.Register(context.Background(), node, true)).To(Succeed())

			// Load a model and give it in-flight requests so it cannot be evicted
			Expect(registry.SetNodeModel(context.Background(), node.ID, "busy-model", 0, "loaded", "", 0, "")).To(Succeed())
			Expect(registry.IncrementInFlight(context.Background(), node.ID, "busy-model", 0)).To(Succeed())

			router := NewSmartRouter(registry, SmartRouterOptions{DB: db})

			start := time.Now()
			_, err := router.evictLRUAndFreeNode(context.Background())
			elapsed := time.Since(start)

			Expect(err).To(MatchError(ErrEvictionBusy))
			// 5 retries * 500ms = 2.5s nominal; allow generous upper bound
			Expect(elapsed).To(BeNumerically("<", 5*time.Second))
		})

		It("respects context cancellation", func() {
			node := &BackendNode{
				Name:     "cancel-evict",
				NodeType: NodeTypeBackend,
				Address:  "10.0.0.101:50051",
			}
			Expect(registry.Register(context.Background(), node, true)).To(Succeed())
			Expect(registry.SetNodeModel(context.Background(), node.ID, "cancel-model", 0, "loaded", "", 0, "")).To(Succeed())
			Expect(registry.IncrementInFlight(context.Background(), node.ID, "cancel-model", 0)).To(Succeed())

			router := NewSmartRouter(registry, SmartRouterOptions{DB: db})

			ctx, cancel := context.WithCancel(context.Background())
			cancel() // cancel immediately

			start := time.Now()
			_, err := router.evictLRUAndFreeNode(ctx)
			elapsed := time.Since(start)

			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("context cancelled"))
			// Should return very quickly since context is already done
			Expect(elapsed).To(BeNumerically("<", 2*time.Second))
		})
	})

	Describe("stageModelFiles (integration)", func() {
		var (
			db       *gorm.DB
			registry *NodeRegistry
		)

		BeforeEach(func() {
			if runtime.GOOS == "darwin" {
				Skip("testcontainers requires Docker, not available on macOS CI")
			}
			db = testutil.SetupTestDB()
			var err error
			registry, err = NewNodeRegistry(db)
			Expect(err).ToNot(HaveOccurred())
		})

		It("does not mutate the original ModelOptions", func() {
			stager := &fakeFileStager{}
			router := NewSmartRouter(registry, SmartRouterOptions{
				FileStager: stager,
				DB:         db,
			})

			node := &BackendNode{
				ID:      "stage-node-id",
				Name:    "stage-node",
				Address: "10.0.0.200:50051",
			}

			original := &pb.ModelOptions{
				Model:     "test-backend/models/test.gguf",
				ModelFile: "/models/test-backend/models/test.gguf",
				MMProj:    "",
			}

			// Capture original values before staging
			origModel := original.Model
			origModelFile := original.ModelFile
			origMMProj := original.MMProj

			// stageModelFiles creates temp files for os.Stat checks.
			// Since none of our test paths exist on disk, stageModelFiles will
			// skip them (clearing non-existent optional fields). The key property
			// is that the original proto pointer is not modified.
			_, _ = router.stageModelFiles(context.Background(), node, original, "test-model")

			// Verify the original proto was not mutated
			Expect(original.Model).To(Equal(origModel))
			Expect(original.ModelFile).To(Equal(origModelFile))
			Expect(original.MMProj).To(Equal(origMMProj))
		})

		Context("when SharedModelsFilesystem is enabled", func() {
			It("skips EnsureRemote entirely and passes paths through unchanged", func() {
				stager := &fakeFileStager{}
				router := NewSmartRouter(registry, SmartRouterOptions{
					FileStager:             stager,
					DB:                     db,
					SharedModelsFilesystem: true,
				})

				node := &BackendNode{
					ID:      "shared-node-id",
					Name:    "shared-node",
					Address: "10.0.0.201:50051",
				}

				original := &pb.ModelOptions{
					Model:     "test-backend/models/test.gguf",
					ModelFile: "/models/test-backend/models/test.gguf",
					MMProj:    "/models/test-backend/models/mmproj.gguf",
				}

				staged, err := router.stageModelFiles(context.Background(), node, original, "test-model")
				Expect(err).ToNot(HaveOccurred())

				// No staging calls should have been made.
				Expect(stager.ensureCalls).To(BeEmpty())

				// Paths passed through unchanged so the worker reads them
				// from the shared filesystem directly.
				Expect(staged.ModelFile).To(Equal(original.ModelFile))
				Expect(staged.MMProj).To(Equal(original.MMProj))

				// ModelPath set so backends can resolve relative options
				// (vae_path, etc.) against the shared models directory.
				Expect(staged.ModelPath).To(Equal("/models"))
			})

			It("leaves ModelPath unset when ModelFile/Model can't derive frontendModelsDir", func() {
				stager := &fakeFileStager{}
				router := NewSmartRouter(registry, SmartRouterOptions{
					FileStager:             stager,
					DB:                     db,
					SharedModelsFilesystem: true,
				})

				node := &BackendNode{ID: "n", Name: "n", Address: "x:1"}
				// No ModelFile means the suffix-trim heuristic can't run.
				original := &pb.ModelOptions{Model: "test-backend/models/test.gguf"}

				staged, err := router.stageModelFiles(context.Background(), node, original, "test-model")
				Expect(err).ToNot(HaveOccurred())
				Expect(stager.ensureCalls).To(BeEmpty())
				Expect(staged.ModelPath).To(BeEmpty())
			})
		})
	})

	// -----------------------------------------------------------------------
	// narrowByGroupAntiAffinity
	// -----------------------------------------------------------------------
	Describe("narrowByGroupAntiAffinity", func() {
		var (
			reg      *fakeModelRouter
			resolver *fakeConflictResolver
			router   *SmartRouter
			ctx      context.Context
		)

		BeforeEach(func() {
			reg = &fakeModelRouter{}
			resolver = &fakeConflictResolver{conflicts: map[string][]string{}}
			router = NewSmartRouter(reg, SmartRouterOptions{
				ConflictResolver: resolver,
			})
			ctx = context.Background()
		})

		It("returns the input set unchanged when the model has no conflicts", func() {
			candidates := []string{"n1", "n2", "n3"}
			out, err := router.narrowByGroupAntiAffinity(ctx, "lonely", candidates)
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(Equal(candidates))
		})

		It("removes nodes that already host a conflicting model", func() {
			resolver.conflicts["b"] = []string{"a"}
			reg.findNodesWithModelByName = map[string][]BackendNode{
				"a": {{ID: "n1"}},
			}
			out, err := router.narrowByGroupAntiAffinity(ctx, "b", []string{"n1", "n2"})
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(ConsistOf("n2"))
		})

		It("returns the original set unchanged when every candidate has a conflict (soft fallback)", func() {
			resolver.conflicts["b"] = []string{"a"}
			reg.findNodesWithModelByName = map[string][]BackendNode{
				"a": {{ID: "n1"}, {ID: "n2"}},
			}
			candidates := []string{"n1", "n2"}
			out, err := router.narrowByGroupAntiAffinity(ctx, "b", candidates)
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(Equal(candidates))
		})

		It("removes nodes hosting any of multiple conflicting models", func() {
			resolver.conflicts["c"] = []string{"a", "b"}
			reg.findNodesWithModelByName = map[string][]BackendNode{
				"a": {{ID: "n1"}},
				"b": {{ID: "n2"}},
			}
			out, err := router.narrowByGroupAntiAffinity(ctx, "c", []string{"n1", "n2", "n3"})
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(ConsistOf("n3"))
		})

		It("treats a nil candidate set (\"any healthy node\") by returning nil unchanged when narrowing yields nothing", func() {
			resolver.conflicts["b"] = []string{"a"}
			reg.findNodesWithModelByName = map[string][]BackendNode{
				"a": {{ID: "n1"}, {ID: "n2"}},
			}
			out, err := router.narrowByGroupAntiAffinity(ctx, "b", nil)
			Expect(err).ToNot(HaveOccurred())
			// nil in → nil out: caller's "any healthy node" semantics preserved.
			// Hard-narrowing nil would silently exclude every other node.
			Expect(out).To(BeNil())
		})

		It("is a no-op when no resolver is configured", func() {
			plain := NewSmartRouter(reg, SmartRouterOptions{})
			candidates := []string{"n1", "n2"}
			out, err := plain.narrowByGroupAntiAffinity(ctx, "b", candidates)
			Expect(err).ToNot(HaveOccurred())
			Expect(out).To(Equal(candidates))
		})
	})

	Describe("installBackendOnNode singleflight", func() {
		It("coalesces concurrent identical installs into one NATS call", func() {
			node := &BackendNode{ID: "n1", Name: "node-1", Address: "10.0.0.1:50051"}

			// Slow install reply so concurrent calls overlap deterministically.
			started := make(chan struct{}, 5)
			release := make(chan struct{})
			unloader := &fakeUnloader{
				installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:50100"},
			}
			unloader.installHook = func() {
				started <- struct{}{}
				<-release
			}

			router := NewSmartRouter(&fakeModelRouter{}, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: &stubClientFactory{client: &stubBackend{}},
			})

			// Fire 5 concurrent identical installBackendOnNode calls.
			done := make(chan error, 5)
			for i := 0; i < 5; i++ {
				go func() {
					_, err := router.installBackendOnNode(context.Background(), node, "llama-cpp", "my-model", 0, nil)
					done <- err
				}()
			}

			// Only ONE call should have entered the unloader hook (the
			// singleflight leader). The other 4 are coalesced and waiting on
			// the leader's result.
			Eventually(started).Should(Receive())
			Consistently(started, 100*time.Millisecond).ShouldNot(Receive())

			// Release the leader; the other 4 callers receive the same result.
			close(release)
			for i := 0; i < 5; i++ {
				Expect(<-done).ToNot(HaveOccurred())
			}
			Expect(unloader.installCalls).To(HaveLen(1),
				"singleflight should coalesce 5 concurrent identical loads into 1 NATS call")
		})

		It("does NOT coalesce installs for different (modelID, replica) keys", func() {
			node := &BackendNode{ID: "n1", Name: "node-1", Address: "10.0.0.1:50051"}
			unloader := &fakeUnloader{
				installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:50100"},
			}
			router := NewSmartRouter(&fakeModelRouter{}, SmartRouterOptions{
				Unloader:      unloader,
				ClientFactory: &stubClientFactory{client: &stubBackend{}},
			})

			_, err1 := router.installBackendOnNode(context.Background(), node, "llama-cpp", "model-A", 0, nil)
			_, err2 := router.installBackendOnNode(context.Background(), node, "llama-cpp", "model-B", 0, nil)
			_, err3 := router.installBackendOnNode(context.Background(), node, "llama-cpp", "model-A", 1, nil)
			Expect(err1).ToNot(HaveOccurred())
			Expect(err2).ToNot(HaveOccurred())
			Expect(err3).ToNot(HaveOccurred())
			Expect(unloader.installCalls).To(HaveLen(3))
		})
	})
})

// ---------------------------------------------------------------------------
// Fake prefixcache.Provider for SmartRouter prefix-cache routing tests
// ---------------------------------------------------------------------------

type observeRecord struct {
	model string
	chain []uint64
	key   prefixcache.ReplicaKey
}

type invalidateRecord struct {
	model string
	key   prefixcache.ReplicaKey
}

// fakePrefixProvider records all interactions and returns a configurable
// decision.
type fakePrefixProvider struct {
	decideCalls     int
	observed        []observeRecord
	invalidated     []invalidateRecord
	invalidatedNode []string
	decision        prefixcache.PrefixDecision
}

func (f *fakePrefixProvider) Decide(_ string, _ []uint64, _ []prefixcache.ReplicaKey, _ time.Time) prefixcache.PrefixDecision {
	f.decideCalls++
	return f.decision
}

func (f *fakePrefixProvider) Observe(model string, chain []uint64, key prefixcache.ReplicaKey, _ time.Time) bool {
	f.observed = append(f.observed, observeRecord{model: model, chain: append([]uint64(nil), chain...), key: key})
	return true
}

func (f *fakePrefixProvider) Invalidate(model string, key prefixcache.ReplicaKey) {
	f.invalidated = append(f.invalidated, invalidateRecord{model: model, key: key})
}

func (f *fakePrefixProvider) InvalidateNode(model, nodeID string) {
	f.invalidatedNode = append(f.invalidatedNode, model+":"+nodeID)
}

func (f *fakePrefixProvider) Evict(_ time.Time) {}

var _ = Describe("SmartRouter prefix-cache routing", func() {
	var (
		backend  *stubBackend
		factory  *stubClientFactory
		unloader *fakeUnloader
	)

	BeforeEach(func() {
		backend = &stubBackend{healthResult: true}
		factory = &stubClientFactory{client: backend}
		unloader = &fakeUnloader{
			installReply: &messaging.BackendInstallReply{Success: true, Address: "10.0.0.1:9001"},
		}
	})

	// loadedReg builds a fake registry with one loaded healthy replica for
	// "m" on node "X", plus matching replica stats so buildPreference can run.
	loadedReg := func() *fakeModelRouter {
		node := &BackendNode{ID: "X", Name: "node-x", Address: "10.0.0.1:50051"}
		nm := &NodeModel{NodeID: "X", ModelName: "m", Address: "10.0.0.1:9001"}
		return &fakeModelRouter{
			findAndLockNode: node,
			findAndLockNM:   nm,
			getModelScheduling: &ModelSchedulingConfig{
				RoutePolicy: "prefix_cache",
			},
			loadedReplicaStatsByName: map[string][]ReplicaCandidate{
				"m": {{NodeID: "X", InFlight: 0}},
			},
		}
	}

	Context("nil provider (round-robin floor)", func() {
		It("passes a nil preference and never decides or observes", func() {
			reg := loadedReg()
			router := NewSmartRouter(reg, SmartRouterOptions{Unloader: unloader, ClientFactory: factory})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{1, 2, 3})
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())

			Expect(reg.findAndLockPrefs).ToNot(BeEmpty())
			for _, p := range reg.findAndLockPrefs {
				Expect(p).To(BeNil())
			}
		})
	})

	Context("with a provider", func() {
		It("passes the decided node as the preference and observes the pick", func() {
			reg := loadedReg()
			prov := &fakePrefixProvider{decision: prefixcache.PrefixDecision{Hot: prefixcache.ReplicaKey{NodeID: "X"}, HasHot: true, MatchRatio: 1.0}}
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: prov,
				PrefixConfig:   prefixcache.DefaultConfig(),
			})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{1, 2, 3})
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())

			Expect(prov.decideCalls).To(BeNumerically(">=", 1))
			Expect(reg.findAndLockPrefs[0]).ToNot(BeNil())
			Expect(reg.findAndLockPrefs[0].PreferredNodeID).To(Equal("X"))
			Expect(reg.findAndLockPrefs[0].PreferredReplica).To(Equal(0))
			Expect(prov.observed).To(HaveLen(1))
			Expect(prov.observed[0].key).To(Equal(prefixcache.ReplicaKey{NodeID: "X", Replica: 0}))
			Expect(prov.observed[0].chain).To(Equal([]uint64{1, 2, 3}))
		})

		It("routes a recurring prefix back to the previously observed node", func() {
			// Real Index as the provider: first request observes X, second
			// request with the same chain must yield PreferredNodeID == X.
			idx := prefixcache.NewIndex(prefixcache.DefaultConfig())
			reg := loadedReg()
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: idx,
				PrefixConfig:   prefixcache.DefaultConfig(),
			})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{7, 8, 9})
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			// First request landed on X (cold placement on the only candidate)
			// and observed the prefix there.
			dFirst := idx.Decide("m", []uint64{7, 8, 9}, []prefixcache.ReplicaKey{{NodeID: "X", Replica: 0}}, time.Now())
			Expect(dFirst.HasHot).To(BeTrue())
			Expect(dFirst.Hot).To(Equal(prefixcache.ReplicaKey{NodeID: "X", Replica: 0}))

			// Second request, same chain: X is now the warm-cache hot match, so
			// the preference must point at it.
			_, err = router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			last := reg.findAndLockPrefs[len(reg.findAndLockPrefs)-1]
			Expect(last).ToNot(BeNil())
			Expect(last.PreferredNodeID).To(Equal("X"))
			Expect(last.PreferredReplica).To(Equal(0))
		})

		It("prefers the exact hot replica when two replicas share a node", func() {
			// Two replicas of "m" live on the SAME node X: replica 0 and replica
			// 1. A hot prefix observed on (X,0) must produce a preference that
			// locks replica 0 specifically, NOT the sibling replica 1 on the same
			// node. This is the replica-granular regression this change fixes.
			idx := prefixcache.NewIndex(prefixcache.DefaultConfig())
			node := &BackendNode{ID: "X", Name: "node-x", Address: "10.0.0.1:50051"}
			nm := &NodeModel{NodeID: "X", ModelName: "m", ReplicaIndex: 0, Address: "10.0.0.1:9001"}
			reg := &fakeModelRouter{
				findAndLockNode: node,
				findAndLockNM:   nm,
				getModelScheduling: &ModelSchedulingConfig{
					RoutePolicy: "prefix_cache",
				},
				loadedReplicaStatsByName: map[string][]ReplicaCandidate{
					"m": {
						{NodeID: "X", ReplicaIndex: 0, InFlight: 0},
						{NodeID: "X", ReplicaIndex: 1, InFlight: 0},
					},
				},
			}
			// Seed the index so (X,0) is the warm replica for this chain.
			idx.Observe("m", []uint64{1, 2, 3}, prefixcache.ReplicaKey{NodeID: "X", Replica: 0}, time.Now())

			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: idx,
				PrefixConfig:   prefixcache.DefaultConfig(),
			})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{1, 2, 3})
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())

			pref := reg.findAndLockPrefs[0]
			Expect(pref).ToNot(BeNil())
			Expect(pref.PreferredNodeID).To(Equal("X"))
			Expect(pref.PreferredReplica).To(Equal(0),
				"the hot prefix lives on replica 0; the same-node sibling replica 1 must NOT be chosen")
		})

		It("does not decide or observe when no prefix chain is present", func() {
			reg := loadedReg()
			prov := &fakePrefixProvider{decision: prefixcache.PrefixDecision{Hot: prefixcache.ReplicaKey{NodeID: "X"}, HasHot: true, MatchRatio: 1.0}}
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: prov,
				PrefixConfig:   prefixcache.DefaultConfig(),
			})

			_, err := router.Route(context.Background(), "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(prov.decideCalls).To(Equal(0))
			Expect(prov.observed).To(BeEmpty())
			Expect(reg.findAndLockPrefs[0]).To(BeNil())
		})

		It("does not observe for round-robin models even with a chain", func() {
			reg := loadedReg()
			reg.getModelScheduling = &ModelSchedulingConfig{RoutePolicy: "round_robin"}
			prov := &fakePrefixProvider{decision: prefixcache.PrefixDecision{Hot: prefixcache.ReplicaKey{NodeID: "X"}, HasHot: true, MatchRatio: 1.0}}
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: prov,
				PrefixConfig:   prefixcache.DefaultConfig(),
			})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{1, 2, 3})
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(prov.decideCalls).To(Equal(0))
			Expect(prov.observed).To(BeEmpty())
			Expect(reg.findAndLockPrefs[0]).To(BeNil())
		})
	})

	Context("forced-disturb pressure", func() {
		// disturbReg builds a registry with two candidate replicas for "m":
		// the hot node X is saturated (high in_flight) and Y is free. Select
		// will therefore reject the hot node and pick Y, which is the
		// forced-disturb signal. findAndLockNode returns Y so Route succeeds.
		disturbReg := func() *fakeModelRouter {
			nodeY := &BackendNode{ID: "Y", Name: "node-y", Address: "10.0.0.2:50051"}
			nm := &NodeModel{NodeID: "Y", ModelName: "m", Address: "10.0.0.2:9001"}
			return &fakeModelRouter{
				findAndLockNode: nodeY,
				findAndLockNM:   nm,
				getModelScheduling: &ModelSchedulingConfig{
					RoutePolicy: "prefix_cache",
				},
				loadedReplicaStatsByName: map[string][]ReplicaCandidate{
					"m": {{NodeID: "X", InFlight: 50}, {NodeID: "Y", InFlight: 0}},
				},
			}
		}

		It("records pressure when a strong hot match was forced off the warm node", func() {
			reg := disturbReg()
			prov := &fakePrefixProvider{decision: prefixcache.PrefixDecision{
				Hot:        prefixcache.ReplicaKey{NodeID: "X"},
				HasHot:     true,
				MatchRatio: 1.0,
				ColdOrder:  []prefixcache.ReplicaKey{{NodeID: "Y"}, {NodeID: "X"}},
			}}
			pressure := prefixcache.NewPressure(time.Minute)
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: prov,
				PrefixConfig:   prefixcache.DefaultConfig(),
				Pressure:       pressure,
			})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{1, 2, 3})
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())

			Expect(pressure.Count("m", time.Now())).To(BeNumerically(">", 0),
				"hot match existed but the load guard forced us off X: must record pressure")
		})

		It("does not record pressure when the hot node is itself eligible", func() {
			reg := loadedReg() // single node X, in_flight 0 → X stays eligible
			prov := &fakePrefixProvider{decision: prefixcache.PrefixDecision{
				Hot:        prefixcache.ReplicaKey{NodeID: "X"},
				HasHot:     true,
				MatchRatio: 1.0,
			}}
			pressure := prefixcache.NewPressure(time.Minute)
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: prov,
				PrefixConfig:   prefixcache.DefaultConfig(),
				Pressure:       pressure,
			})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{1, 2, 3})
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())

			Expect(pressure.Count("m", time.Now())).To(Equal(0),
				"chosen == hot node, no disturb")
		})

		It("does not record pressure for an all-unique workload with no hot match", func() {
			reg := loadedReg()
			prov := &fakePrefixProvider{decision: prefixcache.PrefixDecision{
				HasHot:     false, // no prefix match at all
				MatchRatio: 0,
				ColdOrder:  []prefixcache.ReplicaKey{{NodeID: "X"}},
			}}
			pressure := prefixcache.NewPressure(time.Minute)
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: prov,
				PrefixConfig:   prefixcache.DefaultConfig(),
				Pressure:       pressure,
			})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{1, 2, 3})
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())

			Expect(pressure.Count("m", time.Now())).To(Equal(0),
				"no hot match means no cache to disturb: must not false-positive")
		})
	})

	Context("removal chokepoint on unload", func() {
		It("removes the replica via the registry so the removal hook invalidates the prefix entry", func() {
			idx := prefixcache.NewIndex(prefixcache.DefaultConfig())
			reg := loadedReg()
			router := NewSmartRouter(reg, SmartRouterOptions{
				Unloader:       unloader,
				ClientFactory:  factory,
				PrefixProvider: idx,
				PrefixConfig:   prefixcache.DefaultConfig(),
			})

			ctx := distributedhdr.WithPrefixChain(context.Background(), []uint64{5, 6})
			// Warm the cache: X now holds the prefix.
			_, err := router.Route(ctx, "m", "models/m.gguf", "llama-cpp", nil, false)
			Expect(err).ToNot(HaveOccurred())
			Expect(idx.Decide("m", []uint64{5, 6}, []prefixcache.ReplicaKey{{NodeID: "X", Replica: 0}}, time.Now()).Hot).To(Equal(prefixcache.ReplicaKey{NodeID: "X", Replica: 0}))

			// UnloadModel must route the eviction through the registry removal
			// chokepoint (RemoveAllNodeModelReplicas). The registry's
			// SetReplicaRemovedHook is what invalidates the prefix index in
			// production; the router no longer invalidates directly. Here the
			// fake registry records the removal but fires no hook, so we assert
			// the chokepoint is exercised rather than the downstream
			// invalidation (covered by the registry hook integration tests).
			Expect(router.UnloadModel(context.Background(), "X", "m")).To(Succeed())
			Expect(reg.removeCalls).To(ContainElement("X:m"),
				"UnloadModel must remove the replica via the registry removal chokepoint")
		})
	})
})
