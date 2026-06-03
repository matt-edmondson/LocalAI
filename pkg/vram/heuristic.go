package vram

import (
	"io/fs"
	"os"
	"path/filepath"
)

// Heuristic constants. Weights are typically already stored in their runtime
// dtype (fp16 safetensors), so VRAM ≈ on-disk weights × a small inflation for
// activations/working buffers, plus a fixed overhead for the CUDA context and
// auxiliary modules (VAE, text encoders). These are deliberately conservative
// seeds; the measured value (a later task) corrects them after the first load.
const (
	heuristicWeightFactor    = 1.2
	heuristicFixedOverhead   = uint64(2) << 30 // 2 GiB
	heuristicMinimumEstimate = uint64(1) << 30 // never return less than 1 GiB for a GPU model
)

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
			if IsWeightFile(p) {
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
