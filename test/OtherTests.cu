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

#include <boost/test/unit_test.hpp>
#include <cuda.h>
#include "gpuShareLib/ContigCopy.cuh"
#include "gpuShareLib/GpuUtils.cuh"
#include "gpuShareLib/Reporter.cuh"

namespace Glucose {
    inline void clearObj(int a) {
        // do nothing
    }
}

#include "gpu/GpuHelpedSolver.h"

namespace GpuShare {

BOOST_AUTO_TEST_SUITE(OtherTest)

__device__ void incrArr(DArr<int> arr) {
    for (int i = 0; i < arr.size(); i++) {
        arr[i] ++;
    }
}

__global__ void dTestIncrease(DArr<int> arr1, DArr<int> arr2) {
    incrArr(arr1);
    incrArr(arr2);
}

__global__ void dTestIncrease(DArr<int> arr) {
    incrArr(arr);
}

__global__ void dTestIncrease(int *v) {
     atomicAdd(v, 1);
}

__global__ void dTestSetAt(int *v, int val) {
    *v = val;
}

BOOST_AUTO_TEST_CASE(testContigCopyArr) {
    StreamPointer sp;
    ContigCopier copier;
    ArrPair<int> ap1 = copier.buildArrPair<int>(4, NULL);
    ArrPair<int> ap2 = copier.buildArrPair<int>(3, NULL);

    BOOST_CHECK_EQUAL(4, ap1.getHArr().size());
    BOOST_CHECK_EQUAL(4, ap1.getDArr().size());
    BOOST_CHECK_EQUAL(3, ap2.getHArr().size());
    BOOST_CHECK_EQUAL(3, ap2.getDArr().size());

    ap1.getHArr()[3] = 2;
    ap2.getHArr()[1] = 7;
    BOOST_CHECK(copier.tryCopyAsync(cudaMemcpyHostToDevice, sp.get()));
    dTestIncrease<<<1, 1, 0, sp.get()>>>(ap1.getDArr(), ap2.getDArr());
    BOOST_CHECK(copier.tryCopyAsync(cudaMemcpyDeviceToHost, sp.get()));
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);

    BOOST_CHECK_EQUAL(3, ap1.getHArr()[3]);
    BOOST_CHECK_EQUAL(8, ap2.getHArr()[1]);
}

BOOST_AUTO_TEST_CASE(testContigCopyDeviceToHostOnly) {
    StreamPointer sp;
    ContigCopier copier;
    ArrPair<int> ap = copier.buildArrPair<int>(1, NULL);
    dTestSetAt<<<1, 1, 0, sp.get()>>>(ap.getDArr().getPtr(), 3);
    BOOST_CHECK(copier.tryCopyAsync(cudaMemcpyDeviceToHost, sp.get()));
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(3, ap.getHArr()[0]);
}


// CUDA says that to read from a pointer of size 4, the address must be a multiple of 4
// Test that we don't get a misalignment if we have sizes 1 and 4
BOOST_AUTO_TEST_CASE(testContigAlignment) {
    StreamPointer sp;
    ContigCopier copier;
    ArrPair<bool> opb = copier.buildArrPair<bool>(1, NULL);
    ArrPair<int> opi = copier.buildArrPair<int>(1, NULL);

    opi.getHArr()[0] = 6;

    BOOST_CHECK(copier.tryCopyAsync(cudaMemcpyHostToDevice, sp.get()));
    dTestIncrease<<<1, 1, 0, sp.get()>>>(opi.getDArr().getPtr());
    BOOST_CHECK(copier.tryCopyAsync(cudaMemcpyDeviceToHost, sp.get()));
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);

    BOOST_CHECK_EQUAL(7, opi.getHArr()[0]);
}

BOOST_AUTO_TEST_CASE(testContigResize) {
    StreamPointer sp;
    ContigCopier copier;
    ArrPair<int> opi = copier.buildArrPair<int>(1, NULL);
    opi.increaseSize(2);

    opi.getHArr()[0] = 6;
    opi.getHArr()[1] = 7;

    BOOST_CHECK(copier.tryCopyAsync(cudaMemcpyHostToDevice, sp.get()));
    dTestIncrease<<<1, 1, 0, sp.get()>>>(opi.getDArr());
    BOOST_CHECK(copier.tryCopyAsync(cudaMemcpyDeviceToHost, sp.get()));
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    printf("s is %d\n", copier.getSize());

    BOOST_CHECK_EQUAL(7, opi.getHArr()[0]);
    BOOST_CHECK_EQUAL(8, opi.getHArr()[1]);
}

__global__ void dClear(DReporter<int> rep) {
    rep.clear();
}

__global__ void dReport(int v, DReporter<int> rep) {
    rep.report(v, getThreadId());
}

BOOST_AUTO_TEST_CASE(RollingReportTestOne) {
    StreamPointer sp;
    ContigCopier cc;
    {
        Reporter<int> rr(cc, sp.get(), 3, 1);
        auto dReporter = rr.getDReporter();

        dClear<<<1, 1, 0, sp.get()>>>(dReporter);
        dReport<<<1, 1, 0, sp.get()>>>(2, dReporter);
        dReport<<<1, 1, 0, sp.get()>>>(5, dReporter);
        exitIfFalse(cc.tryCopyAsync(cudaMemcpyDeviceToHost, sp.get()), POSITION);
        exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
        vec<int> l;
        rr.getCopiedToHost(l);
        BOOST_CHECK_EQUAL(2, l.size());
        BOOST_CHECK_EQUAL(2, l[0]);
        BOOST_CHECK_EQUAL(5, l[1]);
    }
    cc.clear(false);
    {
        Reporter<int> rr(cc, sp.get(), 3, 1);
        auto dReporter = rr.getDReporter();

        dClear<<<1, 1, 0, sp.get()>>>(dReporter);
        dReport<<<1, 1, 0, sp.get()>>>(1, dReporter);
        dReport<<<1, 1, 0, sp.get()>>>(7, dReporter);
        exitIfFalse(cc.tryCopyAsync(cudaMemcpyDeviceToHost, sp.get()), POSITION);
        exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
        vec<int> l;
        rr.getCopiedToHost(l);
        BOOST_CHECK_EQUAL(2, l.size());
        BOOST_CHECK_EQUAL(1, l[0]);
        BOOST_CHECK_EQUAL(7, l[1]);
    }
}

/*
Failing test which checks the destr checks work fine
BOOST_AUTO_TEST_CASE(destrCheckFail) {
    CorrespArr<int> car(4, false, false);
    DArr<int> darr = car.getDArr();
    car.resize(2000, false);
    // we only resize the device once we get a darr
    DArr<int> darr2 = car.getDArr();
    dTestIncrease<<<1, 1>>>(darr);
    exitIfError(cudaDeviceSynchronize(), POSITION);
}
*/

BOOST_AUTO_TEST_SUITE_END()

}

