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
#include "CorrespArr.cuh"
#include <cstdlib>

// one gig
size_t maxPageLockedMem = 1e9;
// #define LOG_MEM

namespace GpuShare {

void assertIsDevicePtr(void *mem) {
#ifndef NDEBUG
    cudaPointerAttributes attributes;
    exitIfError(cudaPointerGetAttributes(&attributes, mem), POSITION);
    // After I upgraded to cuda 11,  attributes.type started to be cudaMemoryTypeUnregistered
    ASSERT_MSG(cudaMemoryTypeDevice == attributes.type || cudaMemoryTypeUnregistered == attributes.type, PRINT(attributes.type));
#endif
}

#ifndef NDEBUG
DestrCheck::DestrCheck() {
    allocMem();
}

void DestrCheck::change() {
    // this will invalidate all the destr check pointers currently pointing
    // to us
    free(hostPtr);
    cudaFree(devPtr);
    allocMem();
}

void DestrCheck::allocMem() {
    hostPtr = new int;
    exitIfError(cudaMalloc(&devPtr, sizeof(int)), POSITION);
    *hostPtr = rand();
    exitIfError(cudaMemcpy(devPtr, hostPtr, sizeof(int), cudaMemcpyHostToDevice), POSITION);
    exitIfError(cudaDeviceSynchronize(), POSITION);
}

DestrCheck::~DestrCheck() {
    *hostPtr = -1;
    exitIfError(cudaMemcpy(devPtr, hostPtr, sizeof(int), cudaMemcpyHostToDevice), POSITION);
    exitIfError(cudaDeviceSynchronize(), POSITION);
    free(hostPtr);
    cudaFree(devPtr);
}

DestrCheckPointer::DestrCheckPointer(const DestrCheck &destrCheck) {
    val = *destrCheck.hostPtr;
    hostPtr = destrCheck.hostPtr;
    devPtr = destrCheck.devPtr;
}

DestrCheckPointer::DestrCheckPointer() {
    val = 0;
    hostPtr = NULL;
    devPtr = NULL;
}

void DestrCheckPointer::check() {
#ifdef __CUDA_ARCH__
    assert(devPtr == NULL || *devPtr == val);
#else
    assert(hostPtr == NULL || *hostPtr == val);
#endif
}

#endif

void printV(cudaMemcpyKind kind) {
    if (kind == cudaMemcpyHostToDevice) {
        printf("cudaMemcpyHostToDevice");
    } else if (kind == cudaMemcpyDeviceToHost) {
        printf("cudaMemcpyDeviceToHost");
    } else if (kind == cudaMemcpyDeviceToDevice) {
        printf("cudaMemcpyDeviceToDevice");
    } else {
        throw;
    }
}

// Contract: capacity in input is always a power of 2
// capacity returned is also a power of 2, greater or equal to newSize
size_t getNewCapacity(size_t capacity, size_t newSize, bool reduceCapacity) {
    while (newSize > capacity) {
        capacity *= 2;
    }
    // If newSize is 0, capacity should be 1
    while (reduceCapacity && newSize * 3 + 1 < capacity) {
        capacity /= 2;
    }
    return capacity;
}

size_t getInitialCapacity(size_t size) {
    size_t cap = 1;
    while (cap < size) {
        cap *= 2;
    }
    return cap;
}

// *pt will point to null if it fails
bool allocMemoryDevice(void **pt, size_t amount) {
    ASSERT_OP(amount, >, 0);
    // I've seen it happening for clause updates in a case where cpu solvers generated lots of clauses
    // and gpu was really slow, so it rarely copied clauses to the gpu so they built up on the cpu
    if (amount > 400000000) {
        printf("Allocating large amount of memory: %zu\n", amount);
    }
    size_t freeMem;
    size_t totalMem;
    exitIfError(cudaMemGetInfo(&freeMem, &totalMem), POSITION);
    if (freeMem < 0.01 * totalMem + amount ) {
        printf("c Little memory left on gpu, refusing to allocate\n");
        printf("c There was %zu left out of %zu, wanted to allocate %zu\n",
            freeMem, totalMem, amount);
        *pt = NULL;
        return false;
    }
    // Even if there's enough free memory, cudaMalloc may fail due to
    // fragmentation
    cudaError_t err = cudaMalloc(pt, amount);
    if (err == cudaErrorMemoryAllocation) {
        printf("Failed to allocate memory on device\n");
        *pt = NULL;
        return false;
    }
    exitIfError(err, POSITION);
#ifdef LOG_MEM
    printf("allocated %zu of device mem at %p\n", amount, *pt);
#endif
    return true;
}

void freeMemoryDevice(void *ptr) {
    #ifdef LOG_MEM
        printf("deallocating device mem at %p\n", ptr);
    #endif
    exitIfError(cudaFree(ptr), POSITION);
}

void* reallocMemoryHost(void *ptr, size_t oldSize, size_t newSize, bool &pageLocked) {
    void* newPtr;
    if (!pageLocked) {
        newPtr = realloc(ptr, newSize);
    }
    else {
        bool oldPageLocked = pageLocked;
        newPtr = allocateMemoryHost(newSize, pageLocked);
        memcpy(newPtr, ptr, min(oldSize, newSize));
        freeMemoryHost(ptr, oldPageLocked);
    }
#ifdef LOG_MEM
    printf("Reallocating on host %p from size %zu to %zu page locked %d new ptr %p\n", ptr, oldSize, newSize, pageLocked, newPtr);
#endif
    assert(newPtr != NULL);
    return newPtr;
}

// *ptr will point to null if it fails
bool reallocMemoryDeviceDontCareAboutValues(void **ptr, size_t oldSize, size_t newSize) {
#ifdef LOG_MEM
    void *oldPtr = *ptr;
#endif
    freeMemoryDevice(*ptr);
    if (!allocMemoryDevice(ptr, newSize)) {
        return false;
    }
#ifdef LOG_MEM
    printf("Reallocating on device without caring about values %p from size %zu to %zu new ptr %p\n", oldPtr, oldSize, newSize, *ptr);
#endif
    return true;
}


// *ptr will NOT point to null if it fails, (nothing will have changed)
bool reallocMemoryDevice(void **ptr, size_t oldSize, size_t newSize) {
    void *newPtr;
    if (!allocMemoryDevice(&newPtr, newSize)) {
        return false;
    }
    // Note: because allocating is synchronous, having an asynchronous memcpy wouldn't make much sense
    exitIfError(cudaMemcpy(newPtr, *ptr, min(oldSize, newSize), cudaMemcpyDeviceToDevice), POSITION);
    exitIfError(cudaFree(*ptr), POSITION);
#ifdef LOG_MEM
    printf("Reallocating on device %p from size %zu to %zu new ptr %p\n", ptr, oldSize, newSize, newPtr);
#endif
    *ptr = newPtr;
    return true;
}

void* allocateMemoryHost(size_t amount, bool &pageLocked) {
    void *hPtr;
    ASSERT_OP(amount, >, 0);
    if (amount >= maxPageLockedMem) {
        pageLocked = false;
        printf("c switching memory from page locked to paged because amount is to high: %zu\n", amount);
    }
    if (pageLocked) {
        // If we allocate 0, then we can't use free on the result
        cudaError_t err = cudaMallocHost(&hPtr, amount);
        if (err != cudaSuccess) {
            pageLocked = false;
            printf("c switching memory from page locked to paged because cudaMallocHost failed, amount was %zu\n", amount);
        }
    } 
    if (!pageLocked) {
        ASSERT_OP(amount, <=, (size_t) 100e9);
        hPtr = malloc(amount);
    }
#ifdef LOG_MEM
        printf("allocated %zu of mem on host at pt %p paged locked %d\n", amount, hPtr, pageLocked);
#endif
    return hPtr;
}

void freeMemoryHost(void *hPtr, bool pageLocked) {
#ifdef LOG_MEM
        printf("freeing mem on host at %p paged locked %d\n", hPtr, pageLocked);
#endif
    if (pageLocked) {
        exitIfError(cudaFreeHost(hPtr), POSITION);
    } else {
        free(hPtr);
    }
}

}