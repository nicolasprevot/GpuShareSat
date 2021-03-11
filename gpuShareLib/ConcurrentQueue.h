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

#ifndef DEF_CONCURRENT_QUEUE
#define DEF_CONCURRENT_QUEUE

#include <atomic>
#include <memory>
#include "Utils.h"
namespace GpuShare {

// this class is NOT thread safe
// It's a like a queue, but all the elements can be accessed via an index
// and this index can grow to large values. We recycle old indexes
// Attention: when an old index is recycled, its value stays the same
template<typename T>
class RingQueue {
    private:
    long minIndex;
    long maxIndex;
    // position in arr for minIndex
    long minPos;
    long size;

    void resize(long newSize) {
        arr.resize(newSize);
        long maxPos = (minPos + maxIndex - minIndex) % size;
        // if maxPos == minPos, we could be empty
        if (maxPos <= minPos && maxIndex > minIndex) {
            long copyAtEndCount = std::min(newSize - size, maxPos);
            copyData(size, 0, copyAtEndCount);
            long remainingCopy = maxPos - copyAtEndCount;
            if (remainingCopy != 0) {
                copyData(0, copyAtEndCount, remainingCopy);
            }
        }
        size = newSize;
    }

    protected:
    std::vector<T> arr;

    virtual void copyData(long to, long from, long number) {
        memmove(&arr[to], &arr[from], number * sizeof(T));
    }

    public:
    RingQueue(const RingQueue &other) = delete;

    RingQueue(long _size):
        minIndex(0),
        maxIndex(0),
        minPos(0),
        size(_size),
        arr(_size) {
            ASSERT_OP(_size, >, 0);
    }

    RingQueue() : RingQueue(1) {
    }

    void setMaxIndex(long newMax) {
        ASSERT_OP(newMax, >=, minIndex);
        ASSERT_OP(newMax, >, maxIndex);
        long minSize = newMax - minIndex;
        if (size < minSize) {
            // resize
            long newSize = size;
            while (newSize < minSize) newSize *= 2;
            resize(newSize);
        }
        maxIndex = newMax;
        assert(maxIndex - minIndex <= size);
    }

    // the address of the object returned is only valid for a short time
    // if we do another get, it may not be valid any more
    T& operator[](long i) {
        ASSERT_OP(i, >=, minIndex);
        ASSERT_OP(i, <, maxIndex);
        return arr[(minPos + i - minIndex) % size];
    }

    // all the data strictly before i will be deleted, and we won't be able
    // to use it again
    void setMinIndex(long i) {
        ASSERT_OP(i, >=, minIndex);
        long oldMinIndex = minIndex;
        minIndex = std::max(minIndex, i);
        maxIndex = std::max(maxIndex, i);
        minPos = (minPos + minIndex - oldMinIndex) % size;
    }

    long getSize() const { return size; }

    long getMaxIndex() const { return maxIndex; }
    long getMinIndex() const { return minIndex; }

};

// This class is thread safe
// It is intended for recycling: ie when an element isn't needed any more, it's memory will be
// recycled for next use
// There is one index where we add some new
// And there are two indexes where read them
// The reason why we need a pointer is that we return the addresses of elements, and 
// we can resize. Without a pointer, it would change their addresses
// Not using a unique_ptr because RingQueue does a memmove which gives us a warning on a unique_ptr
template<typename T>
class ConcurrentQueue : public RingQueue<T*> {
private:
    // Elements available to be returned are from RingQueue.minPos (inclusive) to maxIndex (exclusive)
    long interIndex;
    long maxInd;// this is RingQueue.maxIndex or RingQueue.maxIndex - 1
    std::mutex lock;

protected:
    void copyData(long to, long from, long number) {
        long maxTo = to + number;
        while (to < maxTo) {
	    std::swap(RingQueue<T*>::arr[to], RingQueue<T*>::arr[from]);
            to++;
            from++;
        }
    }

public:

    ConcurrentQueue(long _size):
        RingQueue<T*>(_size),
        interIndex(0),
        maxInd(0) {
        assert(_size >= 2);
    }

    ConcurrentQueue(const ConcurrentQueue &other) = delete;

    // The returned item won't be available to the other threads yet
    T& getNew() {
        std::lock_guard<std::mutex> lockGuard(lock);
        ASSERT_OP(maxInd, ==, RingQueue<T*>::getMaxIndex());
        RingQueue<T*>::setMaxIndex(maxInd + 1);
        T*& ptr = (*this)[maxInd];
        if (ptr == NULL) {
            ptr = new T();
        }
        return *ptr;
    }

    // Adds it so it will be available to the other threads
    void addNew() {
        std::lock_guard<std::mutex> lockGuard(lock);
        ASSERT_OP(maxInd + 1, ==, RingQueue<T*>::getMaxIndex());
        maxInd++;
    }

    bool getIncrInter(T*& t) {
        std::lock_guard<std::mutex> lockGuard(lock);
        if (maxInd != interIndex) {
            t = &(*(*this)[interIndex]);
            interIndex ++;
            return true;
        }
        return false;
    }

    // The first one that was added and hasn't been removed
    bool getMin(T*& t) {
        std::lock_guard<std::mutex> lockGuard(lock);
        long minIndex = RingQueue<T*>::getMinIndex();
        if (interIndex != minIndex) {
            t = &(*(*this)[minIndex]);
            return true;
        }
        return false;
    }

    // Remove it
    void removeMin() {
        std::lock_guard<std::mutex> lockGuard(lock);
        long minIndex = RingQueue<T*>::getMinIndex();
        ASSERT_OP(minIndex, !=, interIndex);
        RingQueue<T*>::setMinIndex(minIndex + 1);
    }

    ~ConcurrentQueue() {
        std::vector<T*>& arr = RingQueue<T*>::arr;
        for (unsigned int i = 0; i < arr.size(); i++) {
            if (arr[i] != NULL) delete arr[i];
        }
    }

};
}

#endif
