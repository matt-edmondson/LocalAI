package nodes

import "sort"

// effectiveFree is a GPU's free VRAM minus its in-tick soft reservation.
func effectiveFree(g NodeGPU) uint64 {
	if g.ReservedVRAM >= g.FreeVRAM {
		return 0
	}
	return g.FreeVRAM - g.ReservedVRAM
}

// planGPUSet decides which physical GPU indices on a single node should host a
// model needing `estimate` bytes. Policy:
//   - Best-fit on ONE GPU: the smallest effective-free GPU that still fits
//     (preserves whole GPUs for models that actually need them).
//   - Else span the FEWEST GPUs (largest effective-free first) whose combined
//     effective-free covers the estimate.
//   - Returns (nil, false) when the node cannot fit the model.
//
// Returned indices are sorted ascending for deterministic assignment.
func planGPUSet(gpus []NodeGPU, estimate uint64) ([]int, bool) {
	if estimate == 0 || len(gpus) == 0 {
		return nil, false
	}

	// 1. Best-fit single GPU.
	bestIdx := -1
	var bestFree uint64
	for _, g := range gpus {
		ef := effectiveFree(g)
		if ef >= estimate && (bestIdx == -1 || ef < bestFree) {
			bestIdx = g.GPUIndex
			bestFree = ef
		}
	}
	if bestIdx != -1 {
		return []int{bestIdx}, true
	}

	// 2. Span fewest GPUs: take largest effective-free first until covered.
	sorted := make([]NodeGPU, len(gpus))
	copy(sorted, gpus)
	sort.Slice(sorted, func(i, j int) bool { return effectiveFree(sorted[i]) > effectiveFree(sorted[j]) })

	var acc uint64
	var chosen []int
	for _, g := range sorted {
		ef := effectiveFree(g)
		if ef == 0 {
			continue
		}
		acc += ef
		chosen = append(chosen, g.GPUIndex)
		if acc >= estimate {
			sort.Ints(chosen)
			return chosen, true
		}
	}
	return nil, false
}

// largestFreeGPU returns the index of the GPU with the most effective-free
// VRAM (>0). Used as the liveness fallback when the model's VRAM estimate is
// unknown (0): we can't size it, but we can still pin it to ONE GPU (fixing the
// cuda:0 pile-up) instead of failing the load. Returns (_, false) when no GPU
// has any effective-free VRAM.
func largestFreeGPU(gpus []NodeGPU) (int, bool) {
	bestIdx := -1
	var bestFree uint64
	for _, g := range gpus {
		ef := effectiveFree(g)
		if ef > 0 && (bestIdx == -1 || ef > bestFree) {
			bestIdx = g.GPUIndex
			bestFree = ef
		}
	}
	if bestIdx == -1 {
		return 0, false
	}
	return bestIdx, true
}
