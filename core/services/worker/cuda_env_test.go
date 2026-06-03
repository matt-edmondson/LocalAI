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
