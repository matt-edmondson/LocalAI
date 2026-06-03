package nodes

import "testing"

func gpu(idx int, free uint64) NodeGPU {
	return NodeGPU{GPUIndex: idx, FreeVRAM: free, TotalVRAM: 12 << 30}
}

func TestPlanGPUSet(t *testing.T) {
	const GiB = uint64(1) << 30
	cases := []struct {
		name     string
		gpus     []NodeGPU
		estimate uint64
		want     []int
		wantOK   bool
	}{
		{"fits one, best-fit picks smallest-fitting", []NodeGPU{gpu(0, 11*GiB), gpu(1, 7*GiB)}, 6 * GiB, []int{1}, true},
		{"one full one free lands on free", []NodeGPU{gpu(0, 200*1<<20), gpu(1, 11*GiB)}, 7 * GiB, []int{1}, true},
		{"needs spanning, fewest GPUs", []NodeGPU{gpu(0, 8*GiB), gpu(1, 8*GiB), gpu(2, 8*GiB)}, 15 * GiB, []int{0, 1}, true},
		{"does not fit anywhere", []NodeGPU{gpu(0, 4*GiB), gpu(1, 4*GiB)}, 20 * GiB, nil, false},
		{"reserved reduces effective free", []NodeGPU{{GPUIndex: 0, FreeVRAM: 11 * GiB, ReservedVRAM: 6 * GiB}}, 6 * GiB, nil, false},
		{"zero estimate does not fit (liveness handled by caller)", []NodeGPU{gpu(0, 11*GiB)}, 0, nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := planGPUSet(tc.gpus, tc.estimate)
			if ok != tc.wantOK {
				t.Fatalf("ok=%v want %v", ok, tc.wantOK)
			}
			if !intSliceEqual(got, tc.want) {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}

func TestLargestFreeGPU(t *testing.T) {
	const GiB = uint64(1) << 30
	idx, ok := largestFreeGPU([]NodeGPU{gpu(0, 2*GiB), gpu(1, 9*GiB), gpu(2, 5*GiB)})
	if !ok || idx != 1 {
		t.Fatalf("largestFreeGPU = (%d,%v), want (1,true)", idx, ok)
	}
	if _, ok := largestFreeGPU(nil); ok {
		t.Fatalf("largestFreeGPU(nil) ok=true, want false")
	}
	if _, ok := largestFreeGPU([]NodeGPU{{GPUIndex: 0, FreeVRAM: 1 * GiB, ReservedVRAM: 1 * GiB}}); ok {
		t.Fatalf("largestFreeGPU with no effective-free ok=true, want false")
	}
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
