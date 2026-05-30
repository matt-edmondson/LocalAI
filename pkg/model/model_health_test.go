package model

import "testing"

// A remote model that keeps failing its health check must accumulate failures
// so the loader can evict it after the threshold, while a model that recovers
// (MarkHealthy) resets the counter so a transient timeout never evicts it.
func TestRecordHealthFailureAccumulatesAndResets(t *testing.T) {
	m := NewModel("m", "127.0.0.1:50051", nil)

	if got := m.RecordHealthFailure(); got != 1 {
		t.Fatalf("first failure count = %d, want 1", got)
	}
	if got := m.RecordHealthFailure(); got != 2 {
		t.Fatalf("second failure count = %d, want 2", got)
	}

	// A successful health check must clear the streak.
	m.MarkHealthy()
	if got := m.RecordHealthFailure(); got != 1 {
		t.Fatalf("failure count after MarkHealthy = %d, want 1 (counter should reset)", got)
	}
}

// A persistently-dead remote model reaches the eviction threshold within
// remoteHealthFailureEvictThreshold consecutive failures.
func TestRecordHealthFailureReachesEvictThreshold(t *testing.T) {
	m := NewModel("m", "127.0.0.1:50051", nil)

	var count int
	for i := 0; i < remoteHealthFailureEvictThreshold; i++ {
		count = m.RecordHealthFailure()
	}
	if count != remoteHealthFailureEvictThreshold {
		t.Fatalf("after %d failures, count = %d, want %d",
			remoteHealthFailureEvictThreshold, count, remoteHealthFailureEvictThreshold)
	}
}
