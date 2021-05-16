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
#include <stdlib.h>

#include <cuda.h>
#include <boost/test/unit_test.hpp>
#include "../gpuShareLib/CorrespArr.cuh"
#include "../gpuShareLib/GpuUtils.cuh"
#include "testUtils/TestHelper.cuh"

namespace GpuShare {

BOOST_AUTO_TEST_SUITE( CorrespArrTest )

#define N 3
#define ROW 3
#define COL 3

__device__ int copiedData[N];
__device__ int copiedDataMat[ROW][COL];

__global__ void dTestCopy(DArr<int> arr) {
    for (int i = 0; i < N; i++) {
        copiedData[i] = arr[i];
    }
}

__global__ void dTestIncrease(DArr<int> ptr) {
    for (int i = 0; i < ptr.size(); i++) {
        ptr[i] ++;
    }
}

__global__ void dTestIncreaseOne(int *ptr) {
    (*ptr) ++;
}

void runAndCopy(CorrespArr<int> &cra, int arr[N]) {
    dTestCopy<<<1, 1>>>(cra.getDArr());
    cudaMemcpyFromSymbol(arr, copiedData, N * sizeof(int));
}

void testResizeHelper(bool pagedLocked) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cra(4, pagedLocked, logger);
    BOOST_CHECK_EQUAL(4, cra.size());
    cra.resize(7, true);
    BOOST_CHECK_EQUAL(7, cra.size());
    BOOST_CHECK_EQUAL(7, cra.getDArr().size());
    cra[6] = 8;
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    dTestIncrease<<<1, 1, 0, sp.get()>>>(cra.getDArr());
    cra.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(9, cra[6]);
    cra.resize(3, true);
    BOOST_CHECK_EQUAL(3, cra.size());
}

BOOST_AUTO_TEST_CASE(testResize) {
    testResizeHelper(true);
    // testResizeHelper(false);
}

BOOST_AUTO_TEST_CASE(testValuesKeptWhenIncreaseSize) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cra(1, true, logger);
    cra[0] = 5;
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    cra[0] = 2;
    cra.resize(6, true);
    dTestIncrease<<<1, 1, 0, sp.get()>>>(cra.getDArr());
    cra.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(6, cra[0]);
}

BOOST_AUTO_TEST_CASE(testValuesKeptWhenDecreaseSize) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cra(19, true, logger);
    cra[1] = 3;
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    cra[1] = 1;
    cra.resize(2, true);
    dTestIncrease<<<1, 1, 0, sp.get()>>>(cra.getDArr());
    cra.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(4, cra[1]);
}

BOOST_AUTO_TEST_CASE(testDecreaseSizeThenCopy) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cra(19, true, logger);
    cra[1] = 3;
    cra.resize(2, true);
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    dTestIncrease<<<1, 1, 0, sp.get()>>>(cra.getDArr());
    cra.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(4, cra[1]);
}

// copy to the device
BOOST_AUTO_TEST_CASE(testCopyAllToDevice) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cra(N, true, logger);
    cra[0] = 3;
    cra[1] = 2;
    cra[2] = 2;
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    int arr[N];
    runAndCopy(cra, arr);
    BOOST_CHECK_EQUAL(3, arr[0]);
    BOOST_CHECK_EQUAL(2, arr[1]);
    BOOST_CHECK_EQUAL(2, arr[2]);
}

// copy only some values to the device
BOOST_AUTO_TEST_CASE(testCopySomeToDevice) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cra(N, true, logger);
    cra.setAllTo(2);
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    cra[0] = 3;
    cra[1] = 7;
    cra[2] = 6;
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get(), 1, 2);
    int arr[N];
    runAndCopy(cra, arr);
    BOOST_CHECK_EQUAL(2, arr[0]);
    BOOST_CHECK_EQUAL(7, arr[1]);
    BOOST_CHECK_EQUAL(2, arr[2]);
}

BOOST_AUTO_TEST_CASE(testCopyAllToHost) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cra(N, true, logger);
    cra[0] = 3;
    cra[1] = 7;
    cra[2] = 6;
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    dTestIncrease<<<1,1, 0, sp.get()>>>(cra.getDArr());
    cra.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(4, cra[0]);
    BOOST_CHECK_EQUAL(8, cra[1]);
    BOOST_CHECK_EQUAL(7, cra[2]);
}

BOOST_AUTO_TEST_CASE(testResizeDontCareAboutCurrent) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cra(2, true, logger);
    cra[0] = 3;
    cra[1] = 7;
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    dTestIncrease<<<1,1, 0, sp.get()>>>(cra.getDArr());
    cra.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    cra.resize(3, true);
    cra[2] = 4;
    cra.copyAsync(cudaMemcpyHostToDevice, sp.get());
    dTestIncrease<<<1,1, 0, sp.get()>>>(cra.getDArr());
    cra.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(5, cra[0]);
    BOOST_CHECK_EQUAL(9, cra[1]);
    BOOST_CHECK_EQUAL(5, cra[2]);
}

BOOST_AUTO_TEST_CASE(testAllocTooMuchMem) {
    Logger logger {2, directPrint};
    ArrAllocator<int> aa(4, logger);
    BOOST_CHECK(!aa.tryResize(10000000000, false, false));
    BOOST_CHECK(aa.tryResize(10, true, false));
    int* pt = aa.getDevicePtr();
}

BOOST_AUTO_TEST_CASE(testSetAllTo0) {
    StreamPointer sp;
    Logger logger {2, directPrint};
    CorrespArr<int> cro(2, true, logger);
    cro.getDArr().setAllTo0();
    cro.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(0, cro[0]);
}

// Commented because this test is expected to fail. It's used to make sure that the destr check code is failing when it should
/*
BOOST_AUTO_TEST_CASE(failIfUsingDeletedOnHost) {
    MinHArr<int> minHArr;
    {
        CorrespArr<int> carr(4);
        minHArr = carr.asMinHArr();
    }
    minHArr[2] = 3;
}
*/


// Commented because this test is expected to fail. It's used to make sure that the destr check code is failing when it should
/*
BOOST_AUTO_TEST_CASE(failIfUsingDeletedOnDevice) {
    DArr<int> dArr;
    {
        CorrespArr<int> carr(4, false);
        dArr = carr.getDArr();
    }
    dTestIncrease<<<1,1>>>(dArr);
    exitIfError(cudaDeviceSynchronize(), POSITION);
}
*/

BOOST_AUTO_TEST_SUITE_END()

}


