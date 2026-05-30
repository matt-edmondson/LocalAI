package model

import (
	"sync"
	"time"

	grpc "github.com/mudler/LocalAI/pkg/grpc"
	process "github.com/mudler/go-processmanager"
)

// healthCheckTTL is the duration for which a successful health check is cached.
// Subsequent checkIsLoaded calls within this window skip the gRPC round-trip,
// avoiding serialization of concurrent requests behind ml.mu.Lock().
const healthCheckTTL = 30 * time.Second

// remoteHealthFailureEvictThreshold is how many consecutive failed health
// checks a remote/distributed model may accumulate before it is evicted from
// the in-memory cache and force-reloaded. A single timeout can mean "busy"
// rather than "dead", so we tolerate a few; a genuinely gone backend (worker
// rolled, OOM, crash) keeps timing out and trips the threshold, instead of
// being served as a phantom forever.
const remoteHealthFailureEvictThreshold = 3

type Model struct {
	ID                        string `json:"id"`
	address                   string
	client                    grpc.Backend
	process                   *process.Process
	lastHealthCheck           time.Time
	consecutiveHealthFailures int
	sync.Mutex
}

func NewModel(ID, address string, process *process.Process) *Model {
	return &Model{
		ID:      ID,
		address: address,
		process: process,
	}
}

// NewModelWithClient creates a Model with a pre-configured gRPC client.
// Used in distributed mode where the client is wrapped with file staging.
func NewModelWithClient(ID, address string, client grpc.Backend) *Model {
	return &Model{
		ID:      ID,
		address: address,
		client:  client,
	}
}

func (m *Model) Process() *process.Process {
	return m.process
}

// IsRecentlyHealthy returns true if the model passed a health check within the TTL.
func (m *Model) IsRecentlyHealthy() bool {
	m.Lock()
	defer m.Unlock()
	return !m.lastHealthCheck.IsZero() && time.Since(m.lastHealthCheck) < healthCheckTTL
}

// MarkHealthy records the current time as the last successful health check
// and clears the consecutive-failure counter.
func (m *Model) MarkHealthy() {
	m.Lock()
	defer m.Unlock()
	m.lastHealthCheck = time.Now()
	m.consecutiveHealthFailures = 0
}

// RecordHealthFailure increments and returns the count of consecutive failed
// health checks since the last successful one. Used to distinguish a
// transiently-busy remote model (recovers, resetting via MarkHealthy) from a
// genuinely dead one (keeps failing) so the latter can be evicted and reloaded
// instead of cached forever.
func (m *Model) RecordHealthFailure() int {
	m.Lock()
	defer m.Unlock()
	m.consecutiveHealthFailures++
	return m.consecutiveHealthFailures
}

func (m *Model) GRPC(parallel bool, wd *WatchDog) grpc.Backend {
	if m.client != nil {
		return m.client
	}

	enableWD := false
	if wd != nil {
		enableWD = true
	}

	m.Lock()
	defer m.Unlock()
	m.client = grpc.NewClient(m.address, parallel, wd, enableWD)
	return m.client
}
