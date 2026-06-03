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
	if _, hasVRAM := body["available_vram"]; hasVRAM {
		if _, hasGPUs := body["gpus"]; !hasGPUs {
			t.Fatalf("available_vram present but gpus missing: %#v", body)
		}
	}
}
