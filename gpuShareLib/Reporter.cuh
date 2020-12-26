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

#ifndef REPORTER_CUH_
#define REPORTER_CUH_

#include "CorrespArr.cuh"
#include "ContigCopy.cuh"
#include "GpuUtils.cuh"
#include "Helper.cuh"
#include <algorithm>
#include <functional>

namespace GpuShare {

template<typename T>
class DReporter {
private:
    DArr<T> toReport;
    int countPerCategory;
    DArr<int> pos;

public:
    __device__ void report(T val, int threadId) {
        ASSERT_OP(pos.size(), >, 0);
        int category = threadId % pos.size();
        int p = atomicAdd(&(pos[category]), 1);
        ASSERT_OP(p, >=, countPerCategory * category);
        if (p < countPerCategory * (category + 1)) {
            toReport[p] = val;
        }
    }

    // expects all threads in the kernel to call this
    __device__ void clear() {
        int min, max;
        assignToThread(pos.size(), min, max);
        for (int cat = min; cat < max; cat++) {
            pos[cat] = countPerCategory * cat;
        }
    }

    int getCountPerCategory() { return countPerCategory; }

    DReporter(DArr<T> _toReport, int _countPerCategory, DArr<int> _pos):
        toReport(_toReport),
        countPerCategory(_countPerCategory),
        pos(_pos) {
    }
};

// This class is used so that the device reports something to the host
// It will report a maximum number of items each run.
// Nothing gets copied from the host to the device, only from the device to the host
// Issue is that we don't know how many things will be reported in advance
// So pick a size arbitrarily. If we notice that we didn't have enough size to report
// everything, double the size
// some reports can be missed
// the point is to do only one copy from the device to the host.
template<typename T>
class Reporter {
private:
    ArrPair<int> posPair;
    ArrPair<T> reportPair;
    bool isDone;
    int countPerCategory;

public:
    Reporter(ContigCopier &contigCopier, cudaStream_t &stream, int _countPerCategory, int categoryCount):
        // current contigCopier could already be in use for a rolling reporter, and something could be scheduled
        // to be copied to cpu so we need to sync the stream before resizing
        reportPair(contigCopier.buildArrPair<T>(_countPerCategory * categoryCount, &stream)),
        posPair(contigCopier.buildArrPair<int>(categoryCount, &stream)),
        isDone(false),
        countPerCategory(_countPerCategory) {
            ASSERT_OP(categoryCount, >=, 1);
            ASSERT_OP(countPerCategory, >=, 1);
    }

    DReporter<T> getDReporter() {
        return DReporter<T>(reportPair.getDArr(), countPerCategory, posPair.getDArr());
    }

    // assumes that the contig copier has copied things to the host
    // returns if we should double the count per category
    bool getCopiedToHost(vec<T> &res) {
        assert(!isDone);
        isDone = true;
        bool doubleSize = false;
        int totalCount = 0;
        MinHArr<int> hostPos = posPair.getHArr();
        MinHArr<T> hostReport = reportPair.getHArr();
        for (int cat = 0; cat < hostPos.size(); cat++) {
            int c = min(hostPos[cat] - cat * countPerCategory, countPerCategory);
            ASSERT_OP_MSG(c, >=, 0, PRINT(cat); PRINT(hostPos[cat]); PRINT(countPerCategory));
            if (c >= 0.9 * countPerCategory) {
                doubleSize = true;
            }
            totalCount += c;
        }
        res.resize(totalCount);
        int current = 0;
        for (int cat = 0; cat < hostPos.size(); cat++) {
            int c = min(hostPos[cat] - cat * countPerCategory, countPerCategory);
            if (c > 0) memcpy(res.getData() + current, hostReport.getPtr() + cat * countPerCategory, c * sizeof(T));
            current += c;
        }
        return doubleSize;
    }
};

} /* namespace GpuShare */


#endif /* ROLLINGREPORTER_CUH_ */

