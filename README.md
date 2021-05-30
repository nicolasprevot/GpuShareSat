# GpuShareSat

## General description

GpuShareSat is a solver for the boolean satisfiability problem (SAT).
It is a portfolio solver based upon glucose-syrup which uses both the CPU and the GPU via CUDA. It solves 22 more instances of the SAT 2020 competition than glucose-syrup. The CPU runs a multithreaded portfolio conflict driven SAT solver (like glucose syrup).

In traditional portfolio SAT solvers, each CPU thread export clauses it learns directly to other threads. It uses a heuristic (size, lbd...) to determine which clauses are good enough to share. 

In contrast, in GpuShareSat, a thead will only import a clause if this clause would have been useful in the past few milliseconds (in which case it is likely to be useful again soon).

This is done by exporting all learned clauses to the GPU. The GPU checks its clauses against past partial assignments coming from the CPU threads. This allows the GPU to notice when a clause would have been useful for a CPU thread. In this case, that CPU thread gets notified and imports the clause. 

The GPU repeatedly checks up to millions of clauses against up to 1024 assignments. Experiments show that the GPU is more than able to cope with assignments coming from the CPU (provided the CPU only sends the parent of a conflict).

## Directory overview:

- glucose-syrup:  The glucose-syrup solver with GpuShareSat
- rel-newtech:    The Relaxed LCMDCBDL newTech with GpuShareSat
- gpuShareLib:    The GpuShare Sat library itself

## To build
First, follow [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) if you don't have CUDA installed yet
```
cd glucose-syrup/gpu
make
```
or 
```
cd rel-newtech/gpu
make
```

## Usage:

in glucose-syrup/gpu directory: ./glucose-gpu --help

in rel-newtech/gpu directory: ./rel-newtech-gpu --help

## Contact
[nicolas.prevt@gmail.com](mailto:nicolas.prevt@gmail.com)
