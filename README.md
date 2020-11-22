# GpuShareSat

GpuShareSat is a portfolio SAT solver based upon glucose-syrup using both the CPU and the GPU via CUDA. It solves 22 more instances of the SAT 2020 competition than glucose-syrup. The CPU runs a multithreaded portfolio conflict driven SAT solver (like glucose syrup). However, CPU threads do not export clauses they learn directly to each other. Instead, they export these clauses to the GPU. The GPU checks its clauses against past partial assignments coming from the CPU threads. This allows the GPU to notice when a clause would have been useful for a CPU thread. In this case, that CPU thread gets notified and imports the clause. 

The GPU repeatedly checks up to millions of clauses against up to 1024 assignments. Experiments show that the GPU is more than able to cope with assignments coming from the CPU (provided the CPU only sends the parent of a conflict).

## Directory overview:

- mtl:            Minisat Template Library. Utilities not related to SAT.
- satUtils:       Some utilities related to SAT.
- core:           A core version of the solver glucose (no main here)
- simp:           An extended solver with simplification capabilities
- gpu:            Involves the gpu
- test:           Some unit tests
- perftest:       Some performance tests involving the gpu

## To build
release version: without assertions, statically linked, etc   
Like minisat....

cd  simp  
make rs

To compile the GPU version:
- Follow [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) if you don't have CUDA installed yet
- cd gpu
- make 

## Usage:

in simp directory: ./glucose --help

in gpu directory: ./glucose-gpu --help

## General design of the GPU version:
Regarding threads: 
I've tried to keep the cpu threads as separate as possible. That is, each one has data that only it will read / write.
There are a few places where several threads can read / write. In this case, it will be protected by a lock. 

There are two types:
- the solver threads
- the gpu caller thread: deals directly with the gpu, tells the solver threads about the reported clauses...

To index: ctags **/*.cu **/*.cuh **/*.cc **/*.h
