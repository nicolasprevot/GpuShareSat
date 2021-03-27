/***************************************************************************************
 GpuShareSat -- Copyright (c) 2020, Nicolas Prevot

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 **************************************************************************************************/
#include <stdio.h>
#include "Helper.cuh"
#include "AssertC.cuh"
#include <assert.h>
#include <algorithm>

void exitIfError(cudaError_t err, const char pos[]) {
    if (err != cudaSuccess) {
        fprintf(stderr, "kernel_error in %s: %s\n", pos, cudaGetErrorString(err));
        THROW();
    }
}

void exitIfFalse(bool val, const char pos[]) {
    if (!val) {
        printf("Error in %s\n", pos);
        THROW();
    }
}

__device__ void assignToThread(int count, int threadId, int threadCount, int &min, int &max) {
    uint countPerThread = (count - 1) / threadCount + 1;
    min = threadId * countPerThread;
    max = (threadId + 1) * countPerThread;
    // this can happen. For example, if we have 100 threads and count is 50, 
    // countPerThread will be 1
    if (min > count) min = count;
    if (max > count) max = count;
}

__device__ void assignToThread(int count, int &min, int &max) {
    assignToThread(count, blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x, min, max);
}

__device__ int getThreadId() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

void exitIfLastError(const char pos[]) {
    exitIfError(cudaGetLastError(), pos);
}

namespace GpuShare {

__device__ __host__ void printC(long v) {
    printf("%ld ", v);
}

__device__ __host__ void printC(unsigned long v) {
    printf("%lu ", v);
}

__device__ __host__ void printC(void* pt) {
    printf("%p ", pt);
}

__device__ __host__ void printC(int v) {
    printf("%d ", v);
}

__device__ __host__ void printC(unsigned int v) {
    printf("%u ", v);
}

__device__ __host__ int getRequired(int a, int b) {
    // division of negative numbers is different
    if (a == 0) return 0;
    assert(a > 0);
    int c = (a - 1) / b + 1;
    assert(b * c >= a);
    return c;
}

__device__ __host__ void printBinaryDH(uint x) {
    for (int i = 31; i >= 0; i--) {
        if ((1 << i) & x) {
            printf("1");
        }
        else printf("0");
    }
}

void runGpuAdjustingDims(int &warpsPerBlockGuideline, int totalWarps, std::function<void (int blockCount, int threadsPerBlock)> func) {
    assert(warpsPerBlockGuideline > 0);
    while (true) {
        int warpsPerBlock = std::min(warpsPerBlockGuideline, totalWarps);
        func(getRequired(totalWarps, warpsPerBlockGuideline), warpsPerBlock * WARP_SIZE);
        cudaError_t err = cudaGetLastError();
        if (err == cudaErrorInvalidConfiguration || err == cudaErrorLaunchOutOfResources) {
            int newWarpsPerBlockCount = (int) (0.9 * warpsPerBlockGuideline - 1);
            printf("Got error %s when launching the GPU, decreasing the number of warps per block from %d to %d. Total warps was %d\n", cudaGetErrorString(err), warpsPerBlockGuideline, newWarpsPerBlockCount, totalWarps);
            assert(newWarpsPerBlockCount > 0);
            warpsPerBlockGuideline = newWarpsPerBlockCount;
        } else {
            exitIfError(err, POSITION);
            return;
        }
    }
}

}
