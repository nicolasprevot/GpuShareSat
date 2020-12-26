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
/*
 * Utils.cuh
 *
 *  Created on: 2 Jun 2018
 *      Author: nicolas
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_

namespace GpuShare {

// The point of this class is to not have to manually create / delete the cudaStream_t object
// It's generally a good practice to encapsulate memory allocation / deallocation
class StreamPointer {
public:
    StreamPointer();
    ~StreamPointer();
    cudaStream_t& get() {return stream; }

private:
    cudaStream_t stream;
};

// The point of this class is to not have to manually create / delete the cudaEvent_t object
// It's generally a good practice to encapsulate memory allocation / deallocation
class EventPointer {
public:
    EventPointer();
    ~EventPointer();
    cudaEvent_t& get() {return event; }

private:
    cudaEvent_t event;
};

struct GpuDims {
    int blockCount;
    int threadsPerBlock;

    GpuDims(int _blockCount, int _threadsPerBlock) {
        blockCount = _blockCount;
        threadsPerBlock = _threadsPerBlock;
    }

    int totalCount() {
        return blockCount * threadsPerBlock;
    }
};

}

#endif /* UTILS_CUH_ */
