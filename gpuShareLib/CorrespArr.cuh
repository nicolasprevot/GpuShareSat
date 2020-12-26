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

// This class aims at making memory allocation on host / device safer than using raw pointers directly

#ifndef DEF_CORRESP_ARR
#define DEF_CORRESP_ARR
#include <exception>
#include "Helper.cuh"
#include "utils/Assert.h"
#include "BaseTypes.cuh"
#include "GpuUtils.cuh"

// #define LOG_MEM

namespace GpuShare {

void printV(cudaMemcpyKind kind);

#ifdef LOG_MEM
#define CUDA_MEMCPY_ASYNC(addr1, addr2, amount, mode, stream)\
printf("Copying %zu mem to %p from %p with kind ", amount, (void*)(addr1), (void*)(addr2));\
printV(mode);\
printf("\n");\
exitIfError(cudaMemcpyAsync(addr1, addr2, amount, mode, stream), POSITION);
#else
#define CUDA_MEMCPY_ASYNC(addr1, addr2, amount, mode, stream)\
exitIfError(cudaMemcpyAsync(addr1, addr2, amount, mode, stream), POSITION);
#endif


// DestrCheck and DestrCheckPointer are used so that if someoene tries to use some memory that
// has been deallocated, it fails
#ifndef NDEBUG
class DestrCheck;
class DestrCheckPointer {
    private:
    int *hostPtr;
    int *devPtr;
    int val;
    public:
    __device__ __host__ DestrCheckPointer(const DestrCheck &destrCheck);
    __device__ __host__ DestrCheckPointer();
    __device__ __host__ void check();
    __device__ __host__ bool pointsToSomething() { return hostPtr != NULL; }
    __device__ __host__ void clear() { hostPtr = NULL; devPtr = NULL; }
};

class DestrCheck {
    private:
    int *hostPtr;
    int *devPtr;
    void allocMem();

    public:
    DestrCheck();
    void change();
    ~DestrCheck();
    friend DestrCheckPointer;
};
#endif

// Asserts that there is memory allocated to this specific pointer on the device. Fails if it has been freed
void assertIsDevicePtr(void *mem);

typedef unsigned int uint;

void printV(uint);
__device__ void printVD(size_t);

// ATTENTION: Nothing in this file invoke construtor / destructor on elements on the host (or the device)

bool allocMemoryDevice(void **pt, size_t amount);

void freeMemoryDevice(void *ptr);

void* reallocMemoryHost(void *ptr, size_t oldSize, size_t newSize, bool &pageLocked);

bool reallocMemoryDevice(void **ptr, size_t oldSize, size_t newSize);

bool reallocMemoryDeviceDontCareAboutValues(void **ptr, size_t oldSize, size_t newSize); 

void* allocateMemoryHost(size_t amount, bool &pageLocked);

void freeMemoryHost(void *hPtr, bool pageLocked);

size_t getNewCapacity(size_t currentCapacity, size_t newSize, bool reduceCapacity);

size_t getInitialCapacity(size_t size);

// The point of this being a macro is to get an error message with the line number
#define CHECK_POS(pos, arr) \
    ASSERT_OP(pos, >=, 0);\
    ASSERT_OP(pos, <, arr.size());


// Note: constructors, destructors... are not invoked
template <class T>
class DArr{
private:
    T *_d_ptr;
    size_t _size;
#ifndef NDEBUG
    DestrCheckPointer _destrCheckPointer;
#endif

public:
    // avoid using this
    __device__ __host__ T* getPtr() {return _d_ptr;}

    __device__ __host__ DArr(): DArr(0, NULL
#ifndef NDEBUG
    , DestrCheckPointer()
#endif    
    ) {
    }

#ifndef NDEBUG
    __device__ __host__ DArr(size_t size, T *d_ptr, DestrCheckPointer destrCheckPointer) : _destrCheckPointer(destrCheckPointer)
#else
    __device__ __host__ DArr(size_t size, T *d_ptr)
#endif
     {
        _d_ptr = d_ptr;
        _size = size;
        ASSERT_OP(_size, <, 500000000000);
        if (size > 0) assert(_destrCheckPointer.pointsToSomething());
    }

    __device__ __host__ T *getAddress(size_t p) {
        ASSERT_OP(p, <, _size);
        return _d_ptr + p;
    }

    // template<class T2> void copyArr(DArr<T2> darr, HArr<T2> &arr);

    __device__ T& operator[] (int i) {
        ASSERT_OP(i, >=, 0);
        ASSERT_OP(i, <, _size);
#ifndef NDEBUG
        _destrCheckPointer.check();
#endif
        return _d_ptr[i];
    }

    __device__ __host__ size_t size() {
        return _size;
    }

    __host__ void setToNull() {
        _size = 0;
        _d_ptr = NULL;
#ifndef NDEBUG
        _destrCheckPointer = DestrCheckPointer();
#endif
    }

    __host__ void setAllTo0() {
        cudaMemset(_d_ptr, 0, _size * sizeof(T));
    }

    template<typename T2> DArr<T2> __host__ __device__ getSubArr(size_t start, size_t size) {
        ASSERT_OP_MSG(start * sizeof(T) + size * sizeof(T2), <=, _size * sizeof(T), PRINT(start); PRINT(size); PRINT(_size));
        return DArr<T2>(size, (T2*)&(_d_ptr[start])
#ifndef NDEBUG
         , _destrCheckPointer
#endif
         );
    }
};

template<typename T> void printV(DArr<T> darr) {
    printf("DArr: { size: %ld, addr: %p}", darr.size(), darr.getPtr());
}

template<typename T> __global__ void setAllTo(DArr<T> darr, T value) {
    int min, max;
    assignToThread(darr.size(), min, max);
    for (int i = min; i < max; i++) {
        darr[i] = value;
    }
}

template<typename T> void initDArr(DArr<T> darr, T value, int &warpsPerBlock, int totalWarps) {
    runGpuAdjustingDims(warpsPerBlock, totalWarps, [&] (int blockCount, int threadsPerBlock) {
        setAllTo<<<blockCount, threadsPerBlock>>>(darr, value);
    });
    exitIfError(cudaDeviceSynchronize(), POSITION);
}

// Represents an array, we just have the size and a pointer to the first element
// doesn't handle allocating / deallocating
template <class T>
class MinHArr {

protected:
    T *_h_ptr;
    size_t _size;
#ifndef NDEBUG
    DestrCheckPointer _destrCheckPointer;
    // here to make sure that we don't write on the cpu while there's still
    // a copy in progress, since it would make this copy invalid
    cudaEvent_t *_copyDone;
#endif


public:
    MinHArr() {
        _size = 0;
        _h_ptr = NULL;
#ifndef NDEBUG
        _copyDone = NULL;
#endif
    }

    T* getPtr() {
        assert(_h_ptr != NULL); 
        return _h_ptr;
    } 

    MinHArr(size_t size, T *h_ptr) {
        _h_ptr = h_ptr;
        _size = size;
#ifndef NDEBUG
        _copyDone = NULL;
#endif
    }


#ifndef NDEBUG
    MinHArr(size_t size, T *h_ptr
    , DestrCheckPointer destrCheckPointer
    , cudaEvent_t *copyDone
    )
    : _destrCheckPointer(destrCheckPointer),
    _copyDone(copyDone)
         {
        _h_ptr = h_ptr;
        _size = size;
    }
#endif

    MinHArr<T> withSize(int newSize) {
        ASSERT_OP(newSize, <=, _size);
        return MinHArr<T>(newSize, _h_ptr
#ifndef NDEBUG
        , _destrCheckPointer
        , _copyDone
#endif
        );
    }

    void clear() {
        _size = 0;
        _h_ptr = NULL;
#ifndef NDEBUG
        _copyDone = NULL;
#endif
    }

    void operator=(const MinHArr &other) {
        _h_ptr = other._h_ptr;
        _size = other._size;
#ifndef NDEBUG
        _destrCheckPointer = other._destrCheckPointer;
        _copyDone = other._copyDone;
#endif
    }

#ifndef NDEBUG
// We are slightly more restrictive than we could. If there's a copy to the gpu going on, reading on the cpu is fine, but we'll crash
    void checkEvent() {
        if (_copyDone != NULL) {
            cudaError_t err = cudaEventQuery(*_copyDone);
            if (err == cudaErrorNotReady) {
                // special error message for this case
                fprintf(stderr, "Trying to access a MinHArr when there's still a copy to the gpu in progress. Type size is %ld type is %s\n", sizeof(T), typeid(T).name());
                THROW();
            } else {
                exitIfError(err, POSITION);
            }
            _copyDone = NULL;
        }
    }
#endif

    T& operator[] (size_t i) {
#ifndef NDEBUG
        checkEvent();
        _destrCheckPointer.check();
#endif
        ASSERT_OP(i, <, _size);
        return getPtr()[i];
    }

    size_t size() const {
        return _size;
    }

    void setPtr(T *h_ptr) {
        _h_ptr = h_ptr;
    }

    void setSize(size_t size) {
        _size = size;
    }

    T *getAddress(size_t p) {
        ASSERT_OP(p, <, MinHArr<T>::_size);
        return getPtr() + p;
    }

    template<typename T2> MinHArr<T2> getSubArr(size_t start, size_t size) {
        ASSERT_OP(start * sizeof(T) + size * sizeof(T2), <=, _size * sizeof(T));
        return MinHArr<T2>(size, (T2*)&(getPtr()[start])
#ifndef NDEBUG
        , _destrCheckPointer
        , _copyDone
#endif
        );
    }

    bool isNull() { 
        return _h_ptr == NULL;
    }

    void setAllTo(T val) {
        for (int i = 0; i < size(); i++) {
            operator[](i) = val;
        }
    }

};

template<typename T> void printV(MinHArr<T> harr) {
    printf("MinHArr: { size: %ld, addr: %p}", harr.size(), harr.getPtr());
}

template<typename T>
T getSum(MinHArr<T> arr) {
    T res = 0;
    for (size_t i = 0; i < arr.size(); i++) {
        res += arr[i];
    }
    return res;
}

// Objects can be realloced directly
// If things are copied to the device, it will be a straight memcpy, though
template <class T>
class HArr : public MinHArr<T> {
    private:
        bool _pageLocked;
        size_t _capacity;
#ifndef NDEBUG
        DestrCheck _destrCheck;
#endif

        void reallocAndChangeCapacity(int newSize, bool reduceCapacity) {
            size_t newCapacity = getNewCapacity(_capacity, newSize, reduceCapacity);
            if (newCapacity != _capacity) {
#ifndef NDEBUG
                MinHArr<T>::checkEvent();
                _destrCheck.change();
#endif
                MinHArr<T>::_h_ptr = (T*) reallocMemoryHost((void*)MinHArr<T>::getPtr(), _capacity * sizeof(T),
                        newCapacity * sizeof(T), _pageLocked);
                _capacity = newCapacity;
            }
        }

    public:

        HArr(bool pageLocked):
            MinHArr<T>(0, (T*) allocateMemoryHost(1 * sizeof(T), pageLocked)
#ifndef NDEBUG
            , DestrCheckPointer()
            , NULL
#endif
            ),
            _pageLocked(pageLocked),
            _capacity(1) {
        }

        template<class ...Args>
        HArr(size_t size, bool pageLocked):
            MinHArr<T>(size, (T*) allocateMemoryHost(getInitialCapacity(size) * sizeof(T), pageLocked)
#ifndef NDEBUG
            , DestrCheckPointer()
            , NULL
#endif
            ),
            _pageLocked(pageLocked),
            _capacity(getInitialCapacity(size)) {
        }

        HArr(const HArr<T> &other) = delete;
        HArr& operator=(const HArr<T> &other) = delete;

        T& operator[] (size_t i) {
            ASSERT_OP(MinHArr<T>::_size, <=,_capacity);
            return MinHArr<T>::operator[](i);
        }

        // The reason for having this method (given there's resize) is that
        // resize needs arguments that can be used to construct an object,
        // which we don't need if we clear
        void clear(bool reduceCapacity) {
            reallocAndChangeCapacity(0, reduceCapacity);
            MinHArr<T>::_size = 0;
        }

        bool wouldChangeCapacity(size_t newSize, bool reduceCapacity) {
            return getNewCapacity(_capacity, newSize, reduceCapacity) != _capacity;
        }

        template<class ...Args>
        void resize(size_t newSize, bool reduceCapacity) {
            reallocAndChangeCapacity(newSize, reduceCapacity);
            MinHArr<T>::_size = newSize;
        }

        void add(const T &t) {
            resize(MinHArr<T>::_size + 1, false);
            this->operator[](MinHArr<T>::_size - 1) = t;
        }

        ~HArr() {
            freeMemoryHost((void*) MinHArr<T>::getPtr(), _pageLocked);
        }

        MinHArr<T> asMinHArr() {
            return MinHArr<T>(MinHArr<T>::_size, MinHArr<T>::getPtr()
#ifndef NDEBUG
            , DestrCheckPointer(_destrCheck)
            , NULL
#endif
            );
        }

        int getCapacity() {
            return _capacity;
        }
};

// used to just allocate an array on the device
template <class T> class ArrAllocator {
private:
    T *_d_ptr;
    size_t _size;
    size_t _capacity;
#ifndef NDEBUG
    DestrCheck _destrCheck;
#endif

    // tries to update capacity to have enough for newSize, returns if 
    // succeeded
    bool tryUpdateCapacity(size_t newSize, bool reduceCapacity, bool careAboutValues, cudaStream_t *stream) {
        if (_d_ptr == NULL) {
            if (stream != NULL) {
                exitIfError(cudaStreamSynchronize(*stream), POSITION);
            }
            return tryInitialAllocate(newSize);
        }
        size_t newCapacity = getNewCapacity(_capacity, newSize, reduceCapacity);
        if (newCapacity != _capacity) {
            if (stream != NULL) {
                exitIfError(cudaStreamSynchronize(*stream), POSITION);
            }
            return tryUpdateCapacity(newCapacity, careAboutValues);
        }
        return true;
    }

    bool tryInitialAllocate(size_t initialSize) {
        _capacity = getInitialCapacity(initialSize);
        ASSERT_OP(initialSize, <=, _capacity);
#ifdef LOG_MEM
        printf("Allocated capacity %zu size %zu\n", _capacity, initialSize);
#endif
        if (!allocMemoryDevice((void**) &_d_ptr, _capacity * sizeof(T))) {
            _capacity = 0;
            _size = 0;
            return false;
        }
        return true;
    }

    bool tryUpdateCapacity(size_t newCapacity, bool careAboutValues) {
#ifdef LOG_MEM
        printf("Resizing arrAllocator, capacity is %zu, mem is %p\n", _capacity, _d_ptr);
#endif
#ifndef NDEBUG
        _destrCheck.change();
#endif
        if (careAboutValues) {
            if (!reallocMemoryDevice((void**)&_d_ptr, _capacity * sizeof(T), newCapacity * sizeof(T))) {
                return false;
            }
        }
        else {
            if (!reallocMemoryDeviceDontCareAboutValues((void**)&_d_ptr, _capacity * sizeof(T), newCapacity * sizeof(T))) {
                // _d_ptr will be null in this case
                _capacity = 0;
                _size = 0;
                return false;
            }
        }
        _capacity = newCapacity;
#ifdef LOG_MEM
        printf("Done Resizing arrAllocator at %p, capacity is %zu and mem is %p\n", this, _capacity, _d_ptr);
#endif
        return true;
    }

public:
    ArrAllocator(size_t size): _size(size) {
        if (!tryInitialAllocate(size)) {
            THROW_ERROR(printf("Failed to allocate memory on device"));
        }
    }

    DArr<T> getDArr() {
        return DArr<T>(_size, getDevicePtr()
#ifndef NDEBUG 
        , _destrCheck
#endif
        );
    }

    T* getDevicePtr() {
        // Note: _d_ptr will be null if a previous allocation failed
        assert(_d_ptr != NULL);
        assertIsDevicePtr(_d_ptr);
        ASSERT_OP(_size, <=,_capacity);
        return _d_ptr;
    }

    // If stream isn't null: will sync it first if some memory is deallocated
    bool tryResize(size_t newSize, bool reduceCapacity, bool careAboutValues = true, cudaStream_t *stream = NULL) {
        ASSERT_OP(_size, <=, _capacity);
        if (!tryUpdateCapacity(newSize, reduceCapacity, careAboutValues, stream)) {
            return false;
        }
        _size = newSize;
        ASSERT_OP(_size, <=, _capacity); 
        return true;
    }


    size_t size() {
        return _size;
    }

    ArrAllocator(const ArrAllocator&) = delete;

    ~ArrAllocator() {
        cudaFree((void*) _d_ptr);
    }
};

// This class holds a device pointer, a host pointer, and can convert between the two
// ATTENTION: Does not invoke construtor / destructor on elements on the host
template <class T>
class CorrespArr : public HArr<T> {
    private:
    ArrAllocator<T> _darrAllocator;
    bool needToReduceDCapacity;
#ifndef NDEBUG
    EventPointer copyDonePtr;
#endif

    public:

    CorrespArr(bool pageLocked):
        HArr<T>(pageLocked),
        needToReduceDCapacity(false),
        _darrAllocator(0) {
    }

    template<class ...Args>
    CorrespArr(size_t size, bool pageLocked)
        : HArr<T>(size, pageLocked),
          _darrAllocator(size),
          needToReduceDCapacity(false) {
          // Maybe we don't need this memcpy given we already have code which
          // recomputes the device pointer when needed?
        cudaMemcpy(_darrAllocator.getDevicePtr(), MinHArr<T>::getPtr(), size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void copyAsync(cudaMemcpyKind mode, cudaStream_t &stream) {
        exitIfFalse(tryCopyAsync(mode, stream), POSITION);
    }

    void copyAsync(cudaMemcpyKind mode, cudaStream_t &stream, int min, int max) {
        exitIfFalse(tryCopyAsync(mode, stream, min, max), POSITION);
    }

    bool tryCopyAsync(cudaMemcpyKind mode, cudaStream_t &stream) {
        return tryCopyAsync(mode, stream, 0, MinHArr<T>::_size);
    }

    // if stream isn't null: will sync it first is memory is deallocated
    bool tryResizeDeviceToHostSize(bool careAboutCurrentDeviceValues, cudaStream_t *stream = NULL) {
        bool res =  _darrAllocator.tryResize(MinHArr<T>::_size, needToReduceDCapacity, careAboutCurrentDeviceValues, stream);
        if (res) needToReduceDCapacity = false;
        return res;
    }

    // copies between min inclusive and max exclusive
    bool tryCopyAsync(cudaMemcpyKind mode, cudaStream_t &stream, size_t min, size_t max) {
#ifndef NDEBUG
        assertSizes(min, max);
#endif
        size_t amount = (max - min) * sizeof(T);
        if (mode == cudaMemcpyHostToDevice) {
            // In the case where we need to resize the device array: going 
            // to figure out if can forget about all the values in the device
            // array because we'll just copy them from the host. This would
            // make things better because:
            // - it avoids a cudaMemcpyDeviceToDevice
            // - On the device, we can free memory before we allocate the new
            // one which can make a difference if we're low on memory and the
            // size is big
            bool careAboutCurrentDeviceValues = min > 0 || max < MinHArr<T>::_size;
            T *devicePtr;
            if (!tryGetDevicePtr(devicePtr, careAboutCurrentDeviceValues)) {
                return false;
            }
            void *daddr = (void*) (devicePtr + min);
            void* haddr = (void*) (MinHArr<T>::getPtr() + min);
            cudaError_t err = cudaMemcpyAsync(daddr, haddr, amount, mode, stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "Tried copying to device, failed\n");
                fprintf(stderr, "host: %p dev: %p amount: %zu min: %zu cuda error: %s\n", haddr, daddr, amount, min, cudaGetErrorString(err));
                fprintf(stderr, "retries succeeds: %d\n", cudaMemcpyAsync(daddr, haddr, 4, mode, stream) == cudaSuccess);
                THROW();
            }
        } else if (mode == cudaMemcpyDeviceToHost) {
            T *devicePtr;
            if (!tryGetDevicePtr(devicePtr, true)) {
                return false;
            }
            CUDA_MEMCPY_ASYNC((void*) (MinHArr<T>::getPtr() + min), (void*) (devicePtr + min), amount, mode, stream);
        } else {
            throw std::runtime_error("Unknown copying mode " + mode);
        }
#ifndef NDEBUG
        cudaEventRecord(copyDonePtr.get(), stream);
        MinHArr<T>::_copyDone = &(copyDonePtr.get());
#endif
        return true;
    }

    void assertSizes(size_t min, size_t max) {
        ASSERT_OP(max, <=, MinHArr<T>::_size);
        ASSERT_OP(min, <=, max);
    }

    T& operator[] (size_t i) {
        ASSERT_OP(i, <, MinHArr<T>::_size);
        return MinHArr<T>::getPtr()[i];
    }

    bool tryGetDevicePtr(T*& devicePtr, bool careAboutCurrentDeviceValues = true) {
        DArr<T> darr;
        if (!tryGetDArr(darr, careAboutCurrentDeviceValues)) {
            return false;
        }
        devicePtr = darr.getPtr();
        assert(devicePtr != NULL);
        assertIsDevicePtr(devicePtr);
        return true;
    }

    bool tryGetDArr(DArr<T> &dArr, bool careAboutCurrentDeviceValues = true) {
        // If the darr is already at the right size, this will just do nothing.
        // It may not be, though
        bool success = tryResizeDeviceToHostSize(careAboutCurrentDeviceValues);
        if (success) {
            dArr = _darrAllocator.getDArr();
            ASSERT_OP(dArr.size(), ==, MinHArr<T>::_size);
        }
        return success;
    }

    DArr<T> getDArr(bool careAboutCurrentDeviceValues = true) {
        DArr<T> dArr;
        exitIfFalse(tryGetDArr(dArr, careAboutCurrentDeviceValues), POSITION);
        return dArr;
    }

    T* getDevicePtr(bool careAboutCurrentDeviceValues = true) {
        return getDArr(careAboutCurrentDeviceValues).getAddress(0);
    }

    // For the case where they may be something on stream that is currently copying from device to host
    // so if we're going to realloc pointers, sync the stream first, so we don't end up copy to memory
    // which isn't used any more on the host
    template<class ...Args>
    void resizeMaybeSyncStream(int newSize, bool reduceCapacity, cudaStream_t *stream) {
        if (stream != NULL && HArr<T>::wouldChangeCapacity(newSize, reduceCapacity)) {
            exitIfError(cudaStreamSynchronize(*stream), POSITION);
        }
        resize(newSize, reduceCapacity);
    }


    template<class ...Args>
    void resize(size_t newSize, bool reduceCapacity) {
        HArr<T>::resize(newSize, reduceCapacity);
        // Reason for not resizing the device part now: some algorithms may resize several times on the host
        // before needing the device array. We only need to resize on the device when we need it
        needToReduceDCapacity = reduceCapacity || needToReduceDCapacity;
    }

    // This looks like a duplicate of the HArr one. The reason to have it here is to call this class' resize,
    // instead of the HArr one, to also resize the device
    void add(const T &obj) {
        resize(MinHArr<T>::_size + 1, false);
        operator[](MinHArr<T>::_size - 1 ) = obj;
    }

    ~CorrespArr() {
        // the destructor for the base class MinHArr is going to be called
        // after ours. We don't want it to check _copyDone after it has been
        // deleted
#ifndef NDEBUG
        MinHArr<T>::checkEvent();
        MinHArr<T>::_copyDone = NULL;
#endif
    }

};

template<class T> void copyArrAsync(MinHArr<T> harr, DArr<T> darr, cudaStream_t &stream)
{
    ASSERT_OP(harr.size(), ==, darr.size());
    cudaMemcpyAsync(harr.getAddress(0), darr.getAddress(0), harr.size() * sizeof(T),
        cudaMemcpyDeviceToHost, stream);
}

template<typename T> void copy(MinHArr<T> t1, MinHArr<T> t2) {
    ASSERT_OP(t1.size(), >=, t2.size());
    memcpy(t1.getPtr(), t2.getPtr(), t2.size() * sizeof(T));
}

}

#endif
