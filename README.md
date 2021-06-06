# GpuShareSat

## General description

GpuShareSat is a library for the boolean satisfiability problem (SAT). It uses the GPU via CUDA to help different CPU threads to share clauses with each other.

In traditional portfolio SAT solvers, each CPU thread export clauses it learns directly to other threads. It uses a heuristic (size, lbd...) to determine which clauses are good enough to share. 
In contrast, in GpuShareSat, a thead will only import a clause if this clause would have been useful recently (in which case it is likely to be useful again soon).

This is done by exporting all learned clauses to the GPU. The GPU checks its clauses against past partial assignments coming from the CPU threads. This allows the GPU to notice when a clause would have been useful for a CPU thread. In this case, that CPU thread gets notified and imports the clause. 
The GPU repeatedly checks up to millions of clauses against up to 1024 assignments.

## Directory overview:
- glucose-syrup:  The glucose-syrup solver with GpuShareSat
- rel-newtech:    The Relaxed LCMDCBDL newTech solver with GpuShareSat
- gpuShareLib:    The GpuShareSat library itself

## To build
First, follow [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) if you don't have CUDA installed yet
```
cd glucose-syrup/gpu
make glucose-gpu_release
```
or 
```
cd rel-newtech/gpu
make rel-newtech_release
```

## Usage:

in glucose-syrup/gpu directory: ```./glucose-gpu_relase --help```

in rel-newtech/gpu directory: ```./rel-newtech-gpu_release --help```

## Contact
[nicolas.prevt@gmail.com](mailto:nicolas.prevt@gmail.com)
