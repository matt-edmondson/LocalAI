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
