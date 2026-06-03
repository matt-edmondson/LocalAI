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
