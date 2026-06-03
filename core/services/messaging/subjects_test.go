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
