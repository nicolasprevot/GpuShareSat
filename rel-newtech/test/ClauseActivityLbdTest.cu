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
#include "gpuShareLib/Helper.cuh"
#include "gpuShareLib/GpuUtils.cuh"
#include "gpuShareLib/Clauses.cuh"
#include "gpuShareLib/ContigCopy.cuh"
#include "../testUtils/TestHelper.cuh"
#include "../mtl/Vec.h"
#include <math.h>

namespace GpuShare {

__global__ void writeFirstLit(DClauses dClauses, Lit *ptr, int clSize) {
    *ptr = *dClauses.getStartAddrForClause(clSize, 0);
}

struct TestFixture {
    std::vector<unsigned long> globalStats;
    StreamPointer sp;
    Logger logger;
    HostClauses hCls;

    TestFixture(bool actOnly, float actDecay):
        globalStats(100, 0),
        logger {2, directPrint},
        hCls(GpuDims(1, 1), actDecay, actOnly, globalStats, logger) {
    }

    cudaStream_t& getStream() {
        return sp.get();
    }

    void getRemovingLbdAndAct(int &minLimLbd, int &maxLimLbd, float &remAct) {
        std::vector<int> clCountsAtLbd(MAX_CL_SIZE + 1, 0);
        hCls.fillClauseCountsAtLbds(clCountsAtLbd);
        hCls.getRemovingLbdAndAct(minLimLbd, maxLimLbd, remAct, clCountsAtLbd);
    }
};

BOOST_AUTO_TEST_SUITE( ClauseActivityLbdTest )

GpuDims gpuDims(5, 5);

BOOST_AUTO_TEST_CASE(testInitAct) {
    TestFixture tf(false, 1.0);
    HostClauses &hCls = tf.hCls;
    addClause(hCls, {mkLit(0)});
    copyToDeviceAsync(hCls, tf.sp.get(), gpuDims);
    GpuCref gpuCref { 1, 0};

    BOOST_CHECK(fabs(1.0 - hCls.getClauseActivity(gpuCref)) < 0.01);
}

BOOST_AUTO_TEST_CASE(testBumpAct) {
    TestFixture tf(false, 1.0);
    HostClauses& hCls(tf.hCls);

    addClause(hCls, {mkLit(0)});
    GpuCref gpuCref { 1, 0};

    copyToDeviceAsync(hCls, tf.sp.get(), gpuDims);

    hCls.bumpClauseActivity(gpuCref);
    BOOST_CHECK(fabs(2.0 - hCls.getClauseActivity(gpuCref)) < 0.01);
}


BOOST_AUTO_TEST_CASE(testDecay) {
    TestFixture tf(false, 0.5);
    HostClauses& hCls(tf.hCls);

    addClause(hCls, {mkLit(0)});
    GpuCref gpuCref { 1, 0};

    copyToDeviceAsync(hCls, tf.sp.get(), gpuDims);
    BOOST_CHECK(fabs(2.0 - hCls.getClauseActivity(gpuCref)) < 0.01);

    // clauseActIncr is now 2.0
    hCls.decayClauseAct();
    // clauseActIncr is now 4.0

    hCls.bumpClauseActivity(gpuCref);
    BOOST_CHECK(fabs(6.0 - hCls.getClauseActivity(gpuCref)) < 0.01);
}

struct LbdAndAct {
    int lbd;
    // act are normally float but we only support int
    int act;
};

void setLbdAndAct(int lbd, int act, HostClauses &hCls, int &currentLit, cudaStream_t &stream) {
    // A clause cannot have a size strictly smaller than lbd, so set the size to be the lbd
    Logger logger {2, directPrint};
    HArr<Lit> lits(false, logger);
    for (int j = 0; j < lbd; j++) {
        lits.add(mkLit(currentLit++));
    }
    hCls.addClause(lits, lbd);

    copyToDeviceAsync(hCls, stream, gpuDims);
    for (int i = 0; i < act - 1; i++) {
        hCls.bumpClauseActivity(GpuCref { lbd, hCls.getClauseCount(lbd) - 1 });
    }
}

void setLbdsAndActs(std::vector<LbdAndAct> &lbdsAndActs, HostClauses &hCls, cudaStream_t &stream) {
    int currentLit = 0;
    for (int i = 0; i < lbdsAndActs.size(); i++) {
        setLbdAndAct(lbdsAndActs[i].lbd, lbdsAndActs[i].act, hCls, currentLit, stream);
    }
}

void innerTestApproxNthLargestAct(std::vector<int> &act, float expMin, float expMax, int n) {
    TestFixture tf(false, 1.0);
    HostClauses& hCls(tf.hCls);
    std::vector<LbdAndAct> lbdAndActs(0);
    for (int i = 0; i < act.size(); i++) {
        lbdAndActs.push_back(LbdAndAct {3, act[i]});
    }
    setLbdsAndActs(lbdAndActs, hCls, tf.getStream());
    float apprMedian = hCls.approxNthAct(3, 4, n);
    printf("appr median is %f\n", apprMedian);
    BOOST_CHECK(apprMedian > expMin);
    BOOST_CHECK(apprMedian <= expMax);
}

void innerTestRemovingLbdActAct(std::vector<LbdAndAct> &lbdAndActs, float expMin, float expMax, int expMinLimLbd, int expMaxLimLbd, bool actOnly = false) {
    TestFixture tf(actOnly, 1.0);
    HostClauses& hCls(tf.hCls);
    setLbdsAndActs(lbdAndActs, hCls, tf.getStream());
    int minLimLbd;
    int maxLimLbd;
    float remAct;
    tf.getRemovingLbdAndAct(minLimLbd, maxLimLbd, remAct);
    BOOST_CHECK_EQUAL(expMinLimLbd, minLimLbd);
    BOOST_CHECK_EQUAL(expMaxLimLbd, maxLimLbd);
    if (!(remAct > expMin)) {
        printf("Activity is %f, expected it to be strictly greater than %f\n", remAct, expMin);
    }
    BOOST_CHECK(remAct > expMin);
    if (!(remAct <= expMax)) {
        printf("Activity is %f, expected it to be smaller or equal to %f\n", remAct, expMax);
    }
    BOOST_CHECK(remAct <= expMax);
    printf("act is %f exp min is %f exp max is %f\n", remAct, expMin, expMax);
}

BOOST_AUTO_TEST_CASE(testApproxNthLargestAct) {
    std::vector<int> vec(3);
    vec[0] = 1;
    vec[1] = 2;
    vec[2] = 3;
    innerTestApproxNthLargestAct(vec, 2.0, 2.1, 2);
    innerTestApproxNthLargestAct(vec, -0.1, 0.1, 0);
    innerTestApproxNthLargestAct(vec, 3.0, 3.1, 3);
}

BOOST_AUTO_TEST_CASE(testApproxNthLargestActOneBig) {
    std::vector<int> vec(6);
    vec[0] = 1;
    vec[1] = 1;
    vec[2] = 1;
    vec[3] = 3;
    vec[4] = 3;
    vec[5] = 100;
    innerTestApproxNthLargestAct(vec, 1.0, 1.1, 3);
}

BOOST_AUTO_TEST_CASE(testApproxNthLargestActOneVal) {
    std::vector<int> vec(5);
    vec[0] = 1;
    vec[1] = 1;
    vec[2] = 1;
    vec[3] = 1;
    vec[4] = 10;
    innerTestApproxNthLargestAct(vec, 1.0, 1.1, 2);
}

BOOST_AUTO_TEST_CASE(testRemLbdAct) {
    std::vector<LbdAndAct> lbdsActs;
    // better to use pushVec, because if hardcoding the size, we can make mistakes
    lbdsActs.push_back(LbdAndAct { 2, 1 });
    lbdsActs.push_back(LbdAndAct { 2, 2 });
    lbdsActs.push_back(LbdAndAct { 3, 4 });
    lbdsActs.push_back(LbdAndAct { 4, 6 });
    innerTestRemovingLbdActAct(lbdsActs, -0.1, 1.0, 2, 3);
}

BOOST_AUTO_TEST_CASE(testRemLbdAct2) {
    std::vector<LbdAndAct> lbdsActs;
    lbdsActs.push_back(LbdAndAct { 2, 1 });
    lbdsActs.push_back(LbdAndAct { 3, 4 });
    lbdsActs.push_back(LbdAndAct { 3, 9 });
    lbdsActs.push_back(LbdAndAct { 4, 15 });
    innerTestRemovingLbdActAct(lbdsActs, 4.0, 9.0, 3, 4);
}

BOOST_AUTO_TEST_CASE(testDontRemoveLbd2Clauses) {
    std::vector<LbdAndAct> lbdsActs;
    lbdsActs.push_back(LbdAndAct { 2, 1 });
    lbdsActs.push_back(LbdAndAct { 2, 3 });
    lbdsActs.push_back(LbdAndAct { 2, 4 });
    lbdsActs.push_back(LbdAndAct { 2, 5 });
    lbdsActs.push_back(LbdAndAct { 3, 1 });
    innerTestRemovingLbdActAct(lbdsActs, -0.1, 1.0, 2, 3);
}

BOOST_AUTO_TEST_CASE(testActOnly) {
    std::vector<LbdAndAct> lbdsActs;
    lbdsActs.push_back(LbdAndAct { 2, 1 });
    lbdsActs.push_back(LbdAndAct { 8, 5 });
    innerTestRemovingLbdActAct(lbdsActs, 1, 5, 0, 100, true);
}

BOOST_AUTO_TEST_CASE(testActOnlyHugeDifferences) {
    float decay = 0.5;
    TestFixture tf(true, decay);
    HostClauses& hCls(tf.hCls);
    std::vector<Lit> lits(0);
    int currentLit = 0;
    // will have an act of 2
    setLbdAndAct(3, 1, hCls, currentLit, tf.getStream());
    // will have an act of 4
    setLbdAndAct(3, 2, hCls, currentLit, tf.getStream());
    // we want to decay sufficiently to have a large difference but not enough for activities to drop to 0, which won't happen if we don't rescale
    int c = log(RESCALE_CONST / 100) / log ( 1 / decay);
    for (int i = 0; i < c; i++) {
        hCls.decayClauseAct();
    }
    setLbdAndAct(3, 1, hCls, currentLit, tf.getStream());

    int minLimLbd;
    int maxLimLbd;
    float remAct;
    tf.getRemovingLbdAndAct(minLimLbd, maxLimLbd, remAct);
    BOOST_CHECK(remAct > 2);
    BOOST_CHECK(remAct <= 4);
}

BOOST_AUTO_TEST_CASE(testRescale) {
    StreamPointer sp;

    // each decays multiplies activity inc by RESCALE_CONST * 10
    std::vector<unsigned long> globalStats(100, 0);
    Logger logger {2, directPrint};
    HostClauses hCls(GpuDims(1, 1), 1 / (RESCALE_CONST * 10), false, globalStats, logger);
    addClause(hCls, {mkLit(0)});

    GpuCref gpuCref { 1, 0};

    copyToDeviceAsync(hCls, sp.get(), gpuDims);
    BOOST_CHECK(fabs(10 - hCls.getClauseActivity(gpuCref)) < 0.1);

    hCls.decayClauseAct();
    
    hCls.bumpClauseActivity(gpuCref);
    BOOST_CHECK(fabs(100 - hCls.getClauseActivity(gpuCref)) < 0.1);
    hCls.bumpClauseActivity(gpuCref);
    BOOST_CHECK(fabs(200 - hCls.getClauseActivity(gpuCref)) < 0.1);
}

BOOST_AUTO_TEST_SUITE_END()

}
