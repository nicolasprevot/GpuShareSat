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
#ifndef DEF_HELPER_CU
#define DEF_HELPER_CU

#include <pthread.h>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <functional>
#include "my_make_unique.h"

#define WARP_SIZE 32

#define STR_(x) #x
#define STR(x) STR_(x)
#define POSITION STR(__FILE__) ":" STR(__LINE__)

// count have to be assigned to all the threads of a kernel
// min and max define what will be assigned to the current thread
__device__ void assignToThread(int count, int &min, int &max);
__device__ void assignToThread(int count, int threadId, int threadCount, int &min, int &max);

__device__ int getThreadId();

void exitIfError(cudaError_t err, const char pos[]);
void exitIfLastError(const char pos[]);
void exitIfFalse(bool val, const char pos[]);

namespace GpuShare {

__device__ void printVD(long v);
__device__ void printVD(unsigned long v);
__device__ void printVD(int v);
__device__ void printVD(unsigned int v);
__device__ void printVD(void* pt);
__device__ __host__ void printBinaryDH(uint v);

// We generally want to run with many gpu threads per block, but is it difficult to find exactly how many we can run with
// when calling this method, the caller specifies how many threads it wants to run, (and a guideline of how many threads per block)
// If the gpu crashes due to too many threads in a block with these configuration, we retry with fewer threads per block, but
// possibly more blocks. We may end up running with more warps than initially asked, the gpu code we run has to cope with that
void runGpuAdjustingDims(int &warpsPerBlockGuideline, int totalWarps, std::function<void (int blockCount, int threadsPerBlock) > func);

// Number of multiples of b between 0 (included) and a (not included)
// or alternatively: the smallest value c so that b * c >= a
__device__ __host__ int getRequired(int a, int b);

}


#endif
