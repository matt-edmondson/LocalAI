package nodes

import (
	"testing"
	"time"
)

func TestRemoteLoadTimeoutFromEnv(t *testing.T) {
	t.Run("default when unset", func(t *testing.T) {
		t.Setenv("LOCALAI_REMOTE_LOAD_TIMEOUT", "")
		if got := remoteLoadTimeoutFromEnv(); got != defaultRemoteLoadTimeout {
			t.Fatalf("got %v, want %v", got, defaultRemoteLoadTimeout)
		}
	})
	t.Run("valid override", func(t *testing.T) {
		t.Setenv("LOCALAI_REMOTE_LOAD_TIMEOUT", "30m")
		if got := remoteLoadTimeoutFromEnv(); got != 30*time.Minute {
			t.Fatalf("got %v, want 30m", got)
		}
	})
	t.Run("invalid falls back to default", func(t *testing.T) {
		t.Setenv("LOCALAI_REMOTE_LOAD_TIMEOUT", "soon")
		if got := remoteLoadTimeoutFromEnv(); got != defaultRemoteLoadTimeout {
			t.Fatalf("got %v, want %v", got, defaultRemoteLoadTimeout)
		}
	})
	t.Run("non-positive falls back to default", func(t *testing.T) {
		t.Setenv("LOCALAI_REMOTE_LOAD_TIMEOUT", "-1m")
		if got := remoteLoadTimeoutFromEnv(); got != defaultRemoteLoadTimeout {
			t.Fatalf("got %v, want %v", got, defaultRemoteLoadTimeout)
		}
	})
}
