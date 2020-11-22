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

// Doing many small copies between CPU and GPU is slow. So instead, we allocate all the memory we're going
// to copy contiguously. This way, we can pass it with a single cudaMemcpy. This is the goal of this class.


#ifndef CONTIGCOPY_CUH_
#define CONTIGCOPY_CUH_

#include "Helper.cuh"
#include "CorrespArr.cuh"

namespace Glucose {

class ContigCopier;

template<typename T>
class ArrPair {
private:
    int offset;
    int sz;
    ContigCopier *contigCopier;
#ifndef NDEBUG
    DestrCheckPointer destrCheckPointer;
#endif

public:
    ArrPair(int _offset, int _size, ContigCopier &_contigCopier);
    ArrPair();
    // ArrPair &operator=(const ArrPair &other);

    // Note: this value returned shouldn't be kept long, because a resize
    // for another ArrPair of the same ContigCopy would invalidate this one
    MinHArr<T> getHArr();

    // Note: this value returned shouldn't be kept long, because a resize
    // for another ArrPair of the same ContigCopy would invalidate this one
    DArr<T> getDArr();

    bool pointsToSomething() { return contigCopier != NULL; }
    void increaseSize(int newSize);
    int size() { return sz; }

    void reset();
};

// This class is not thread safe
class ContigCopier {
    template <class T2> friend class ArrPair;

private:
    CorrespArr<char> values;
#ifndef NDEBUG
    DestrCheck destrCheck;
#endif

    // Returns the position (in bytes from the start) in the memory
    template <typename T> int reserveMem(int size, cudaStream_t *stream) {
        // we want the next mem which is a multiple of sizeof(T> for alignment reasons
        int res = values.size() + ((-values.size()) % sizeof(T));
        values.resizeMaybeSyncStream(res + size * sizeof(T), false, stream);
        return res;
    }

public:
    ContigCopier(bool pageLocked = false): values(0, pageLocked) {
    }

    void* getHPtr() { return (void*) values.getAddress(0);}
    void* getDPtr() { 
        char* res;
        // This should succeed because it should only be called after
        // we've called copyAsync which would have resized things
        if (!values.tryGetDevicePtr(res, false)) {
            THROW();
        }
        return (void*) res;
    }

    template<typename T>
    std::unique_ptr<ArrPair<T>> buildArrPairPtr(int size, cudaStream_t *stream) {
        int offSet = reserveMem<T>(size, stream);
        return std::make_unique<ArrPair<T>>(offSet, size, *this);
    }

    template<typename T>
    ArrPair<T> buildArrPair(int size, cudaStream_t *stream) {
        int offSet = reserveMem<T>(size, stream);
        return ArrPair<T>(offSet, size, *this);
    }

    bool tryCopyAsync(cudaMemcpyKind kind, cudaStream_t &stream) {
        return values.tryCopyAsync(kind, stream);
    }

    void clear(bool decreaseCapacity) {
        values.resize(0, decreaseCapacity);
    }

    bool tryResizeDeviceToHostSize(bool careAboutCurrentDeviceValues, cudaStream_t *stream) {
        return values.tryResizeDeviceToHostSize(careAboutCurrentDeviceValues, stream);
    }

    int getSize() { 
        return values.size();
    }

    void increaseSize(int newSize) {
        ASSERT_OP(newSize, >=, values.size());
        values.resize(newSize, false);
    }

};

template<typename T>
ArrPair<T>::ArrPair(int _offset, int _size, ContigCopier &_contigCopier):
    offset(_offset),
    sz(_size),
    contigCopier(&_contigCopier)
#ifndef NDEBUG
    , destrCheckPointer(_contigCopier.destrCheck)
#endif
      {
}

template<typename T>
ArrPair<T>::ArrPair() {
    reset();
}

template<typename T>
void ArrPair<T>::reset() {
    offset = 0;
    sz = 0;
    contigCopier = NULL;
}

template<typename T>
void ArrPair<T>::increaseSize(int newSize) {
    assert(contigCopier != NULL);
    ASSERT_OP_MSG(sz * sizeof(T) + offset, ==, contigCopier->getSize(), printf("Can only resize the last arr pair"));
    contigCopier->increaseSize(offset + newSize * sizeof(T));
    sz = newSize;
}

template<typename T> MinHArr<T> ArrPair<T>::getHArr() {
    // if this is called and we then resize the ContigCopy: it will throw thanks to
    // the CorrespArr's destr checks
    assert(pointsToSomething());
    return contigCopier->values.getSubArr<T>(offset, sz);
}

template<typename T> DArr<T> ArrPair<T>::getDArr() {
    assert(pointsToSomething());
    return contigCopier->values.getDArr().getSubArr<T>(offset, sz);
}

}

#endif
