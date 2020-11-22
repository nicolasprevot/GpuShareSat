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
#define BOOST_TEST_MODULE cuda_module
#include <boost/test/unit_test.hpp>
#include "gpu/Helper.cuh"
#include "gpu/GpuUtils.cuh"
#include "gpu/Assigs.cuh"
#include "gpu/Clauses.cuh"
#include "gpu/GpuHelpedSolver.cuh"
#include "gpu/GpuRunner.cuh"
#include "gpu/GpuMultiSolver.cuh"
#include "gpu/Reported.cuh"
#include "satUtils/SolverTypes.h"
#include "core/Solver.h"
#include "testUtils/TestHelper.cuh"
#include "gpu/ContigCopy.cuh"
#include <cuda.h>
#include <mtl/Vec.h>
#include <algorithm>
#include "utils/Utils.h"

#include "gpu/GpuRunner.cuh"

using namespace std;

namespace Glucose {



BOOST_AUTO_TEST_SUITE( GpuSolverTest )

__global__ void dUpdateAssigsGlobal(DValsPerId<VarUpdate> assigUpdates, DArr<DOneSolverAssigs> dOneSolverAssigs, DValsPerId<AggCorresp> dAggCorresps, DAssigAggregates aggregates) {
    dUpdateAssigs(assigUpdates, dOneSolverAssigs, dAggCorresps, aggregates);
}

void updateAssigsAsync(AssigsAndUpdates &assigsAndUpdates, GpuDims gpuDims, cudaStream_t &stream) {
    dUpdateAssigsGlobal<<<gpuDims.blockCount, gpuDims.threadsPerBlock, 0, stream>>>(assigsAndUpdates.dAssigUpdates.get(), assigsAndUpdates.assigSet.dSolverAssigs.getDArr(),
        assigsAndUpdates.assigSet.aggCorresps.get(), assigsAndUpdates.assigSet.dAssigAggregates);
}

AssigsAndUpdates fillAndUpdateAssigs(HostAssigs &hostAssigs, GpuDims gpuDims, ContigCopier &cc, vec<AssigIdsPerSolver> &assigIdsPerSolver, cudaStream_t &stream) {
    cc.clear(false);
    AssigsAndUpdates assigsAndUpdates = hostAssigs.fillAssigsAsync(cc, assigIdsPerSolver, stream);
    exitIfFalse(cc.tryCopyAsync(cudaMemcpyHostToDevice, stream), POSITION);
    updateAssigsAsync(assigsAndUpdates, gpuDims, stream);
    exitIfError(cudaStreamSynchronize(stream), POSITION);
    return assigsAndUpdates;
}


__global__ void dTestAssigs(DArr<DOneSolverAssigs> assigs, int solverId, DArr<MultiLBool> res) {
    for (int i = 0; i < res.size(); i++) {
        res[i] = assigs[solverId].multiLBools[i];
    }
}

// Test that the device can read the assigs
BOOST_AUTO_TEST_CASE(testAssigsTwoSolvers) {
    StreamPointer sp;
    GpuDims gpuDims {2, WARP_SIZE};
    ContigCopier cc; 
    HostAssigs hostAssigs(2, gpuDims);
    int warpsPerBlock = 1;
    hostAssigs.growSolverAssigs(2, warpsPerBlock, 1);

    OneSolverAssigs& assig0 = hostAssigs.getAssigs(0);
    assig0.enterLock();
    BOOST_CHECK(assig0.isAssignmentAvailableLocked());
    assig0.setVarLocked(0, l_False);
    assig0.setVarLocked(1, l_Undef);
    assig0.assignmentDoneLocked();
    assig0.exitLock();

    OneSolverAssigs& assig1 = hostAssigs.getAssigs(1);
    assig1.enterLock();
    BOOST_CHECK(assig1.isAssignmentAvailableLocked());
    assig1.setVarLocked(0, l_Undef);
    assig1.setVarLocked(1, l_True);
    assig1.assignmentDoneLocked();
    assig1.exitLock();
    vec<AssigIdsPerSolver> assigIdsPerSolver;
    AssigsAndUpdates assigsAndUpdates = fillAndUpdateAssigs(hostAssigs, gpuDims, cc, assigIdsPerSolver, sp.get());
    CorrespArr<MultiLBool> res(2, true);

    // solver 0
    dTestAssigs<<<1, 1, 0, sp.get()>>>(assigsAndUpdates.assigSet.dSolverAssigs.getDArr(), 0, res.getDArr());
    res.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(1, res[0].isDef & 1);
    BOOST_CHECK_EQUAL(0, res[0].isTrue & 1);
    BOOST_CHECK_EQUAL(0, res[1].isDef & 1);

    // solver 1
    dTestAssigs<<<1, 1, 0, sp.get()>>>(assigsAndUpdates.assigSet.dSolverAssigs.getDArr(), 1, res.getDArr());
    res.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(0, res[0].isDef & 1);
    BOOST_CHECK_EQUAL(1, res[1].isDef & 1);
    BOOST_CHECK_EQUAL(1, res[1].isTrue & 1);
}

BOOST_AUTO_TEST_CASE(testAssigsTwoAssignments) {
    StreamPointer sp;
    GpuDims gpuDims {2, WARP_SIZE};
    HostAssigs hostAssigs(2, gpuDims);
    int warpsPerBlock = 1;
    hostAssigs.growSolverAssigs(1, warpsPerBlock, 1);

    OneSolverAssigs& assig = hostAssigs.getAssigs(0);
    assig.enterLock();
    BOOST_CHECK(assig.isAssignmentAvailableLocked());
    assig.setVarLocked(0, l_True);
    assig.setVarLocked(1, l_False);
    assig.assignmentDoneLocked();
    assig.exitLock();
    assig.enterLock();
    BOOST_CHECK(assig.isAssignmentAvailableLocked());
    assig.setVarLocked(0, l_Undef);
    assig.assignmentDoneLocked();
    assig.exitLock();
    ContigCopier cc;
    vec<AssigIdsPerSolver> assigIdsPerSolver;
    DArr<DOneSolverAssigs> dAssigs = fillAndUpdateAssigs(hostAssigs, gpuDims, cc, assigIdsPerSolver, sp.get()).assigSet.dSolverAssigs.getDArr();
    CorrespArr<MultiLBool> res(2, true);

    dTestAssigs<<<1, 1, 0, sp.get()>>>(dAssigs, 0, res.getDArr());
    res.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    // first assignment
    BOOST_CHECK_EQUAL(1, res[0].isDef & 1);
    BOOST_CHECK_EQUAL(1, res[0].isTrue & 1);
    BOOST_CHECK_EQUAL(1, res[1].isDef & 1);
    BOOST_CHECK_EQUAL(0, res[1].isTrue & 1);
    // second assignment
    BOOST_CHECK_EQUAL(0, res[0].isDef & 2);
    // this var was set for the first assignment, so it should still be set now
    BOOST_CHECK_EQUAL(2, res[1].isDef & 2);
    BOOST_CHECK_EQUAL(0, res[1].isTrue & 2);
}

BOOST_AUTO_TEST_CASE(testManyAssignments) {
    StreamPointer sp;
    GpuDims gpuDims {2, WARP_SIZE};
    HostAssigs hostAssigs(assigCount(), gpuDims);
    int warpsPerBlock = 1;
    hostAssigs.growSolverAssigs(1, warpsPerBlock, 1);

    OneSolverAssigs& assig = hostAssigs.getAssigs(0);
    for (int i = 0; i < assigCount(); i++) {
        assig.enterLock();
        BOOST_CHECK(assig.isAssignmentAvailableLocked());
        assig.setVarLocked(i, i % 2 == 0 ? l_True : l_False);
        assig.assignmentDoneLocked();
        assig.exitLock();
    }
    assig.enterLock();
    BOOST_CHECK(!assig.isAssignmentAvailableLocked());
    assig.exitLock();
    ContigCopier cc;
    vec<AssigIdsPerSolver> assigIdsPerSolver;
    auto assigsAndUpdates = fillAndUpdateAssigs(hostAssigs, gpuDims, cc, assigIdsPerSolver, sp.get());
    setAllAssigsToLastAsync(warpsPerBlock, 1, assigsAndUpdates, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);

    assig.enterLock();
    BOOST_CHECK(assig.isAssignmentAvailableLocked());
    assig.setVarLocked(7, l_Undef);
    assig.assignmentDoneLocked();
    assig.exitLock();

    DArr<DOneSolverAssigs> dAssigs = fillAndUpdateAssigs(hostAssigs, gpuDims, cc, assigIdsPerSolver, sp.get()).assigSet.dSolverAssigs.getDArr();
    CorrespArr<MultiLBool> res(assigCount(), true);

    dTestAssigs<<<1, 1, 0, sp.get()>>>(dAssigs, 0, res.getDArr());
    res.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);

    // Looking at many variables, but all on the first assignment
    for (int i = 0; i < assigCount(); i++) {
        if (i == 7) {
            BOOST_CHECK_EQUAL(0, res[i].isDef & 1);
        }
        else {
            BOOST_CHECK_EQUAL(1, res[i].isDef & 1);
            BOOST_CHECK_EQUAL(i % 2 == 0, res[i].isTrue & 1);
        }
    }
}

BOOST_AUTO_TEST_CASE(testAssigAggregates) {
    StreamPointer sp;
    GpuDims gpuDims {2, WARP_SIZE};
    HostAssigs hostAssigs(1, gpuDims);
    int warpsPerBlock = 1;
    hostAssigs.growSolverAssigs(2, warpsPerBlock, 1);
    int assigsPerSolver = sizeof(Vals) * 8;
    for (int solv = 0; solv < 2; solv++) {
        OneSolverAssigs& assig = hostAssigs.getAssigs(solv);
        lbool l = solv == 0 ? l_True : l_False;
        for (int i = 0; i < assigsPerSolver; i++) {
            assig.enterLock();
            ASSERT_MSG(assig.isAssignmentAvailableLocked(), PRINT(i); PRINT(solv));
            assig.setVarLocked(0, i % 2 == 0 ? l : l_Undef);
            assig.assignmentDoneLocked();
            assig.exitLock();
        }
    }
    ContigCopier cc;
    vec<AssigIdsPerSolver> assigIdsPerSolver;
    AssigsAndUpdates assigsAndUpdates = fillAndUpdateAssigs(hostAssigs, gpuDims, cc, assigIdsPerSolver, sp.get());
    ASSERT_OP(~((Vals) 0), ==, assigsAndUpdates.assigSet.dAssigAggregates.startVals);
    HArr<MultiAgg> res(1, false);
    BOOST_CHECK_EQUAL(1, assigsAndUpdates.assigSet.dAssigAggregates.multiAggs.size());
    copyArrAsync(res, assigsAndUpdates.assigSet.dAssigAggregates.multiAggs, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL((((Vals) 1) << (assigsPerSolver / 2)) - 1, res[0].canBeTrue);
    BOOST_CHECK_EQUAL(((((Vals) 1) << (assigsPerSolver / 2)) - 1) << (assigsPerSolver / 2), res[0].canBeFalse);
    BOOST_CHECK_EQUAL(~((Vals) 0), res[0].canBeUndef);
}

__global__ void dTestCopyClause(DClauses clauses, DArr<Lit> res) {
    int p = getStartPosForClause(res.size(), 0);
    for (int i = 0; i < res.size(); i++) {
        res[i] = clauses.get(res.size(), p + WARP_SIZE * i);
    }
}

// Tests adding clauses, and that the device can read them
BOOST_AUTO_TEST_CASE(testAddClauseHost) {
    StreamPointer sp;
    CorrespArr<int> clausesCountPerThread(2, true);
    GpuDims gpuDims(2, WARP_SIZE);
    HostClauses hClauses(gpuDims, 0.99, 20, 0, false);
    addClause(hClauses, mkLit(4), mkLit(2));
    CorrespArr<Lit> cra(2, false);

    copyToDeviceAsync(hClauses, sp.get(), gpuDims);

    ContigCopier cc;
    RunInfo runInfo = hClauses.makeRunInfo(sp.get(), cc);
    cc.tryCopyAsync(cudaMemcpyHostToDevice, sp.get());
    dTestCopyClause<<<1, 1, 0, sp.get()>>>(runInfo.getDClauses(), cra.getDArr());
    cra.copyAsync(cudaMemcpyDeviceToHost, sp.get());
    exitIfError(cudaStreamSynchronize(sp.get()), POSITION);
    BOOST_CHECK_EQUAL(mkLit(4).x, cra[0].x);
    BOOST_CHECK_EQUAL(mkLit(2).x, cra[1].x);
}

__global__ void dTestReporter(DReporter<ReportedClause> dreporter) {
    // reports the clause id 0 of size 1 to the assignment 0 and solver 0
    dreporter.report(ReportedClause{1, 0, GpuCref{1, 0}}, getThreadId());
}

template<typename T> __global__ void dClear(DReporter<T> rep) {
    rep.clear();
}

// Tests that the gpu reporter can report wrong clauses and that the cpu can read them
BOOST_AUTO_TEST_CASE(testReported) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 3, 10);
    StreamPointer sp;

    addClause(*fx.co.hClauses, mkLit(3));
    copyToDeviceAsync(*fx.co.hClauses, sp.get(), fx.co.gpuDims);
    BOOST_CHECK_EQUAL(1, fx.co.hClauses->getClauseCount(1));

    int solvId = 0;

    AssigIdsPerSolver aips;
    aips.startAssigId = 0;
    aips.assigCount = 1;
    vec<AssigIdsPerSolver> assigIds(1, aips);

    ContigCopier gpuToCpuCc;
    Reporter<ReportedClause> reporter(gpuToCpuCc, fx.sp.get(), 4, 4);

    auto dReporter = reporter.getDReporter();
    dClear<<<1, 1, 0, fx.sp.get()>>>(dReporter);
    dTestReporter<<< 1, 1, 0, fx.sp.get()>>>(dReporter);

    exitIfFalse(gpuToCpuCc.tryCopyAsync(cudaMemcpyDeviceToHost, fx.sp.get()), POSITION);
    exitIfError(cudaStreamSynchronize(fx.sp.get()), POSITION);
   
    vec<ReportedClause> wcl; 
    reporter.getCopiedToHost(wcl);

    fx.co.reported->fill(assigIds, wcl);

    ClauseBatch *clBatch;
    BOOST_CHECK(fx.co.reported->getIncrReportedClauses(solvId, clBatch));

    MinHArr<Lit> lits;
    GpuClauseId gpuClauseId;
    BOOST_CHECK(clBatch->popClause(lits, gpuClauseId));
    BOOST_CHECK_EQUAL(1, lits.size());
    BOOST_CHECK(lits[0] == mkLit(3));
    BOOST_CHECK_EQUAL(0, gpuClauseId);

    BOOST_CHECK(!clBatch->popClause(lits, gpuClauseId));

    exitIfLastError(POSITION);
}

// There's not a full solver in this test, but everything else
BOOST_AUTO_TEST_CASE(testClausesAssigsReported) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 3, 3);

    addClause(*fx.co.hClauses, mkLit(0));
    addClause(*fx.co.hClauses, mkLit(1));
    addClause(*fx.co.hClauses, mkLit(2));
    copyToDeviceAsync(*fx.co.hClauses, fx.sp.get(), fx.co.gpuDims);

    // assigs for solver 0
    OneSolverAssigs& assig0 = fx.co.hostAssigs->getAssigs(0);
    assig0.enterLock();
    assig0.setVarLocked(0, l_False);
    assig0.setVarLocked(1, l_True);
    assig0.setVarLocked(2, l_True);
    assig0.assignmentDoneLocked();
    assig0.exitLock();

    assig0.enterLock();
    assig0.setVarLocked(0, l_True);
    assig0.setVarLocked(1, l_False);
    assig0.setVarLocked(2, l_False);
    assig0.assignmentDoneLocked();
    assig0.exitLock();

    // assigs for solv 1
    OneSolverAssigs& assig1 = fx.co.hostAssigs->getAssigs(1);
    assig1.enterLock();
    assig1.setVarLocked(0, l_True);
    assig1.setVarLocked(1, l_False);
    assig1.setVarLocked(2, l_True);
    assig1.assignmentDoneLocked();
    assig1.exitLock();
    execute(*fx.co.gpuRunner);
    ASSERT_OP(4, ==, fx.co.reported->getTotalReported());

    ClauseBatch *clBatch;

    MinHArr<Lit> lits;
    GpuClauseId gpuClauseId;

    // solver 0
    BOOST_CHECK(fx.co.reported->getIncrReportedClauses(0, clBatch));
    for (int i = 0 ; i < 3; i++) {
        BOOST_CHECK(clBatch->popClause(lits, gpuClauseId));
    }
    BOOST_CHECK(!clBatch->popClause(lits, gpuClauseId));

    fx.co.reported->removeOldestClauses(0);

    // solver 1: only one clause reported
    BOOST_CHECK(fx.co.reported->getIncrReportedClauses(1, clBatch));
    BOOST_CHECK(clBatch->popClause(lits, gpuClauseId));
    BOOST_CHECK(!clBatch->popClause(lits, gpuClauseId));
    fx.co.reported->removeOldestClauses(1);
   
    // nothing for solver 2 
    BOOST_CHECK(!fx.co.reported->getIncrReportedClauses(2, clBatch));

    assig0.enterLock();
    assig0.setVarLocked(1, l_True);
    assig0.assignmentDoneLocked();
    assig0.exitLock();
    execute(*fx.co.gpuRunner);
    BOOST_CHECK(fx.co.reported->getIncrReportedClauses(0, clBatch));
    BOOST_CHECK(clBatch->popClause(lits, gpuClauseId));
    BOOST_CHECK(!clBatch->popClause(lits, gpuClauseId));

    BOOST_CHECK(!fx.co.reported->getIncrReportedClauses(1, clBatch));
}

// Test that the gpu can read assigs and report the appropriate wrong clause
BOOST_AUTO_TEST_CASE(testFindClausesMultiThread) {
    GpuOptions ops;
    setDefaultOptions(ops);
    ops.blockCount = 1;
    ops.threadsPerBlock = 32;
    GpuFixture fx(ops, 3, 1);

    OneSolverAssigs &oneSolverAssigs = fx.co.hostAssigs->getAssigs(0);
    oneSolverAssigs.enterLock();
    oneSolverAssigs.setVarLocked(0, l_False);
    oneSolverAssigs.setVarLocked(1, l_True);
    oneSolverAssigs.setVarLocked(2, l_Undef);
    oneSolverAssigs.assignmentDoneLocked();
    oneSolverAssigs.exitLock();

    // test copying the clauses several times
    addClause(*fx.co.hClauses, mkLit(0), mkLit(1));
    copyToDeviceAsync(*fx.co.hClauses, fx.sp.get(), fx.co.gpuDims);
    addClause(*fx.co.hClauses, mkLit(0), ~mkLit(1));
    copyToDeviceAsync(*fx.co.hClauses, fx.sp.get(), fx.co.gpuDims);
    addClause(*fx.co.hClauses, ~mkLit(1), mkLit(2));
    copyToDeviceAsync(*fx.co.hClauses, fx.sp.get(), fx.co.gpuDims);

    ContigCopier cc;
    std::unique_ptr<AssigsAndUpdates> assigsAndUpdates;
    std::unique_ptr<Reporter<ReportedClause>> reporter;

    vec<AssigIdsPerSolver> assigIdsPerSolver;
    execute(*fx.co.gpuRunner);
    BOOST_CHECK_EQUAL(2, fx.co.reported->getTotalReported());

    ClauseBatch *clBatch;
    BOOST_CHECK(fx.co.reported->getIncrReportedClauses(0, clBatch));

    MinHArr<Lit> lits1;
    MinHArr<Lit> lits2;
    GpuClauseId gpuClauseId;
    BOOST_CHECK(clBatch->popClause(lits1, gpuClauseId));
    BOOST_CHECK(clBatch->popClause(lits2, gpuClauseId));

    MinHArr<Lit> forCl2;
    MinHArr<Lit> forCl3;

    if (lits1[0] == mkLit(0)) {
        forCl2 = lits1;
        forCl3 = lits2;
    } else {
        forCl2 = lits2;
        forCl3 = lits1;
    }

    BOOST_CHECK_EQUAL(2, forCl2.size());
    BOOST_CHECK_EQUAL(mkLit(0).x, forCl2[0].x);
    BOOST_CHECK_EQUAL((~mkLit(1)).x, forCl2[1].x);

    BOOST_CHECK_EQUAL(2, forCl3.size());
    BOOST_CHECK_EQUAL((~mkLit(1)).x, forCl3[0].x);
    BOOST_CHECK_EQUAL(mkLit(2).x, forCl3[1].x);

    BOOST_CHECK(!clBatch->popClause(lits1, gpuClauseId));
}

BOOST_AUTO_TEST_CASE(SolverImportBinary) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 3, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);
    // level 0: var 0 is true
    solver.uncheckedEnqueue(mkLit(0));
    solver.newDecisionLevel();
    // level 1: var 2 is true
    solver.uncheckedEnqueue(mkLit(2));
    BOOST_CHECK((l_True == solver.value(2)));

    // add gpu clause: 0 implies 1
    addClause(*fx.co.hClauses, ~mkLit(0), mkLit(1));
    fx.executeAndImportClauses();

    solver.propagate();
    BOOST_CHECK((l_True == solver.value(0)));
    BOOST_CHECK((l_True == solver.value(1)));
    // 2 should have been unset
    BOOST_CHECK((l_Undef == solver.value(2)));

    BOOST_CHECK((0 == solver.level(0)));
    BOOST_CHECK((0 == solver.level(1)));
    fx.checkReportedImported(1, 0, false);
}

// The reason to have this test is that it would have failed if it had been there
// The only assig aggregate bits set to the last value were those which were
// used. It needs to be all of them. This test is for this case
BOOST_AUTO_TEST_CASE(testOneAssignmentThenTwo) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 4, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);
    addClause(*fx.co.hClauses, ~mkLit(0), ~mkLit(1), mkLit(2));
    copyToDeviceAsync(*fx.co.hClauses, fx.sp.get(), fx.co.gpuDims);

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));
    solver.copyAssigsForGpu(solver.decisionLevel());
    execute(*fx.co.gpuRunner);

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(3));
    solver.copyAssigsForGpu(solver.decisionLevel());
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));
    solver.copyAssigsForGpu(solver.decisionLevel());
    execute(*fx.co.gpuRunner);
    BOOST_CHECK((1 == fx.co.reported->getTotalReported()));
}


BOOST_AUTO_TEST_CASE(SolverDoesntImportSameClauseTwice) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 3, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);

    // Solver will copy two assignments to gpu, each one will get back the same clause
    // test that the solver only imports the clause once
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));
    solver.copyAssigsForGpu(1);

    solver.cancelUntil(0);
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(~mkLit(1));
    solver.copyAssigsForGpu(1);
    addClause(*fx.co.hClauses, ~mkLit(0), mkLit(1));
    // no need to call copyToDevice because execute does it
    execute(*fx.co.gpuRunner);

    bool foundEmptyClause = false;

    solver.gpuImportClauses(foundEmptyClause);

    BOOST_CHECK_EQUAL(1, fx.co.reported->getTotalReported());
    BOOST_CHECK_EQUAL(1, solver.stats[nbImported]);
}

// Test that if a clause has been imported and then deleted, it can be imported again
BOOST_AUTO_TEST_CASE(SolverCanReimportClause) {
    // In this test: we add two clauses on the gpu, both get imported, then we reduceDb
    // on the cpu, so one gets deleted, and we test that we can import it again
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 5, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);

    // reason for having clauses of size 3: if they were of size 2, they'd be permanently learned so we couldn't test anything
    addClause(*fx.co.hClauses, ~mkLit(0), ~mkLit(1), mkLit(4));
    addClause(*fx.co.hClauses, ~mkLit(0), ~mkLit(1), mkLit(3));

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));
    BOOST_CHECK_EQUAL(0, (int) solver.stats[learnedFromGpu]);
    BOOST_CHECK_EQUAL(0, (int) solver.stats[nbImported]);
    fx.executeAndImportClauses();

    solver.cancelUntil(0);
    BOOST_CHECK_EQUAL(2, (int) solver.stats[learnedFromGpu]);
    solver.reduceDB();
    BOOST_CHECK_EQUAL(1, (int) solver.stats[learnedFromGpu]);
    BOOST_CHECK_EQUAL(2, (int) solver.stats[nbImported]);

    solver.cancelUntil(0);
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));
    // just so that the first clause doesn't need to be imported by the gpu
    solver.propagate();
    fx.executeAndImportClauses();
    BOOST_CHECK_EQUAL(3, (int) solver.stats[nbImported]);
    // the second clause that was removed during the reduceDB has been re-added
    BOOST_CHECK_EQUAL(2, (int) solver.stats[learnedFromGpu]);
}

BOOST_AUTO_TEST_CASE(TwoSolverImportBinary) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 3, 2);

    fx.solvers[0]->uncheckedEnqueue(mkLit(0));
    fx.solvers[1]->uncheckedEnqueue(~mkLit(0));

    fx.solvers[0]->propagate();
    fx.solvers[1]->propagate();

    BOOST_CHECK((l_Undef == fx.solvers[0]->value(1)));
    BOOST_CHECK((l_Undef == fx.solvers[1]->value(1)));

    addClause(*fx.co.hClauses, ~mkLit(0), mkLit(1));
    addClause(*fx.co.hClauses, mkLit(0), ~mkLit(1));

    vec<CRef> v;
    fx.executeAndImportClauses(v);

    fx.solvers[0]->propagate();
    fx.solvers[1]->propagate();

    BOOST_CHECK((l_True == fx.solvers[0]->value(1)));
    BOOST_CHECK((l_False == fx.solvers[1]->value(1)));
}

// Test that assigs on the gpu do get unset
BOOST_AUTO_TEST_CASE(SolverUnsets) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 2, 1);
    addClause(*fx.co.hClauses, mkLit(0), mkLit(1));
    GpuHelpedSolver& solver = *fx.solvers[0];
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));
    vec<CRef> v;
    fx.executeAndImportClauses(v);
    fx.checkReportedImported(0, 0, false);

    solver.cancelUntil(0);
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));

    addClause(*fx.co.hClauses, ~mkLit(0), mkLit(1));
    fx.executeAndImportClauses(v);
    // the clause just added should have been imported because 1 is unset
    fx.checkReportedImported(1, 0, false);
}

BOOST_AUTO_TEST_CASE(SolverImportUnary) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 2, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));

    BOOST_CHECK((l_True == solver.value(1)));

    addClause(*fx.co.hClauses, mkLit(0));
    fx.executeAndImportClauses();
    BOOST_CHECK_EQUAL(1, solver.stats[nbReported]);
    BOOST_CHECK_EQUAL(1, solver.stats[nbImportedUnit]);
    solver.propagate();

    BOOST_CHECK((l_True == solver.value(0)));
    BOOST_CHECK((l_Undef == solver.value(1)));

    BOOST_CHECK((0 == solver.level(0)));
}

BOOST_AUTO_TEST_CASE(SolverHasManyClausesReported) {
    GpuOptions ops;
    setDefaultOptions(ops);
    int varCount = 20;
    GpuFixture fx(ops, varCount, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);


    for (int i = 0; i < varCount; i++) {
        // gpu clause: var is true
        addClause(*fx.co.hClauses, mkLit(i));
        // solver learns clause
        fx.executeAndImportClauses();
        BOOST_CHECK((l_True == solver.value(i)));
    }
    fx.checkReportedImported(varCount, 0, true);
}

BOOST_AUTO_TEST_CASE(SolverHasManyClausesReportedAllAtOnce) {
    GpuOptions ops;
    setDefaultOptions(ops);
    ops.blockCount = 2;
    ops.threadsPerBlock = 32;
    // The point of having that many is to have more than one clause per thread
    int varCount = 4000;
    GpuFixture fx(ops, varCount, 1, 5000);
    GpuHelpedSolver& solver = *(fx.solvers[0]);

    for (int i = 0; i < varCount; i++) {
        // a third of the variables are already true, so the clauses won't be imported
        if (i % 3 == 0) {
            solver.newDecisionLevel();
            solver.uncheckedEnqueue(mkLit(i));
        }
        addClause(*fx.co.hClauses, mkLit(i));
    }
    // solver learns clause
    fx.executeAndImportClauses();
    for (int i = 0; i < varCount; i++) {
        // Those were set at the beginning, so their clauses haven't been imported, and they've
        // been unset because of the other literals added, so they're not set any more
        if (i % 3 == 0) BOOST_CHECK((l_Undef == solver.value(i)));
        else BOOST_CHECK((l_True == solver.value(i)));
    }
    fx.checkReportedImported(varCount - getRequired(varCount, 3), 0, true);
}


BOOST_AUTO_TEST_CASE(OneInstanceTwoAssignments) {
    // in this test: we test that if some clauses become useful at some point,
    // then they will be imported.
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 4, 1);

    GpuHelpedSolver& solver = *(fx.solvers[0]);

    addClause(*fx.co.hClauses, ~mkLit(0), mkLit(2));
    addClause(*fx.co.hClauses, ~mkLit(1), mkLit(3));

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));

    solver.copyAssigsForGpu(solver.decisionLevel());
    solver.cancelUntil(0);
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));
    solver.copyAssigsForGpu(solver.decisionLevel());
    solver.cancelUntil(0);
    BOOST_CHECK_EQUAL(0, fx.co.reported->getTotalReported());
    fx.executeAndImportClauses();

    // at this point, both clauses should have been imported
    // check that when propagating, these clauses are actually used
    BOOST_CHECK_EQUAL(2, fx.co.reported->getTotalReported());
    BOOST_CHECK_EQUAL(2, solver.stats[nbImported]);
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));
    solver.propagate();
    BOOST_CHECK((l_True == solver.value(2)));

    solver.cancelUntil(0);
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));
    solver.propagate();
    BOOST_CHECK((l_True == solver.value(3)));
}

BOOST_AUTO_TEST_CASE(SolverClauseKeptAfterImport) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 3, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));

    addClause(*fx.co.hClauses,  ~mkLit(0), ~mkLit(1), mkLit(2));
    fx.executeAndImportClauses();

    BOOST_CHECK((l_True == solver.value(0)));
    BOOST_CHECK((l_True == solver.value(1)));
    BOOST_CHECK((l_True == solver.value(2)));

    solver.cancelUntil(1); // should cancel 1 and 2
    BOOST_CHECK((l_True == solver.value(0)));
    BOOST_CHECK((l_Undef == solver.value(1)));
    BOOST_CHECK((l_Undef == solver.value(2)));

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));
    solver.propagate();
    BOOST_CHECK((l_True == solver.value(0)));
    BOOST_CHECK((l_True == solver.value(1)));
    BOOST_CHECK((l_True == solver.value(2)));

    fx.checkReportedImported(1, 0, false);
}

BOOST_AUTO_TEST_CASE(SolverImportFalseClauseDifferentLevel) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 3, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);

    solver.addClause(~mkLit(1), mkLit(2));

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));
    solver.propagate();

    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));

    solver.propagate();
    BOOST_CHECK((l_True == solver.value(0)));
    BOOST_CHECK((l_True == solver.value(1)));
    BOOST_CHECK((l_True == solver.value(2)));

    BOOST_CHECK((1 == solver.level(0)));
    BOOST_CHECK((2 == solver.level(1)));
    BOOST_CHECK((2 == solver.level(2)));

    // at this point, 0 and 1 have different levels. So 1 should now just be implied by 0
    addClause(*fx.co.hClauses,  ~mkLit(0), ~mkLit(1));
    fx.executeAndImportClauses();

    BOOST_CHECK((l_True == solver.value(0)));
    BOOST_CHECK((l_False == solver.value(1)));
    BOOST_CHECK((l_Undef == solver.value(2)));

    BOOST_CHECK((1 == solver.level(0)));
    BOOST_CHECK((1 == solver.level(1)));

    fx.checkReportedImported(1, 0, false);
}

BOOST_AUTO_TEST_CASE(testDeduceEmptyClause) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 1, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);

    solver.uncheckedEnqueue(mkLit(0));
    solver.propagate();
    BOOST_CHECK((l_True == solver.value(0)));

    addClause(*fx.co.hClauses, ~mkLit(0));

    fx.execute();
    exitIfLastError(POSITION);
    // we have the clauses v(0) == true and v(0) == false
    BOOST_CHECK((l_False == solver.solve()));
    exitIfLastError(POSITION);
}

BOOST_AUTO_TEST_CASE(findConflict) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 2, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);

    solver.addClause(~mkLit(0), mkLit(1));
    solver.addClause(mkLit(0), ~mkLit(1));
    solver.newDecisionLevel();

    solver.uncheckedEnqueue(mkLit(0));
    solver.propagate();

    BOOST_CHECK_EQUAL(1, solver.level(0));
    BOOST_CHECK_EQUAL(1, solver.level(1));

    addClause(*fx.co.hClauses, ~mkLit(0), ~mkLit(1));
    // we don't call executeAndImportClauses because the import has to be
    // done during solve()
    fx.execute();
    BOOST_CHECK_EQUAL(1, fx.co.hClauses->getClauseCount());
    BOOST_CHECK_EQUAL(1, fx.co.reported->getTimesReported());

    BOOST_CHECK(l_True == solver.solve());
    BOOST_CHECK_EQUAL(1, solver.stats[nbReported]);
    BOOST_CHECK_EQUAL(1, solver.stats[nbImported]);

    BOOST_CHECK(l_False == solver.modelValue(0));
    BOOST_CHECK(l_False == solver.modelValue(1));

    exitIfLastError(POSITION);
}

BOOST_AUTO_TEST_CASE(testReduceDb) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 5, 1);

    addClause(*fx.co.hClauses, ~mkLit(1), mkLit(3), mkLit(4));
    // Adding this clause (the one which will be used) last so that it will have to
    // be passed from one thread to another
    addClause(*fx.co.hClauses, ~mkLit(0), mkLit(2), mkLit(4));

    copyToDeviceAsync(*fx.co.hClauses, fx.sp.get(), fx.co.gpuDims);
    BOOST_CHECK_EQUAL(2, fx.co.hClauses->getClauseCount());
    BOOST_CHECK_EQUAL(6, fx.co.hClauses->getClauseLengthSum());

    GpuHelpedSolver& solver = *(fx.solvers[0]);
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(~mkLit(4));

    vec<CRef> ignored;
    fx.executeAndImportClauses(ignored);
    BOOST_CHECK(l_True == solver.value(2));
    BOOST_CHECK(l_Undef == solver.value(3));

    printf("reduce db\n");
    fx.co.hClauses->reduceDb(fx.sp.get());

    BOOST_CHECK_EQUAL(1, fx.co.hClauses->getClauseCount());
    BOOST_CHECK_EQUAL(3, fx.co.hClauses->getClauseLengthSum());

    // the second clause should have been removed because it wasn't used before
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(1));
    fx.executeAndImportClauses(ignored);
    BOOST_CHECK(l_Undef == solver.value(3));

    int rep = fx.co.reported->getTotalReported();
    // now check that adding more clauses works fine after the reduce db
    addClause(*fx.co.hClauses, ~mkLit(0), ~mkLit(3));
    fx.executeAndImportClauses(ignored);

    BOOST_CHECK_EQUAL(rep + 1, fx.co.reported->getTotalReported());

    BOOST_CHECK(l_False == solver.value(3));
}

BOOST_AUTO_TEST_CASE(testMods) {
    BOOST_CHECK_EQUAL(3, getLargestSameMod(3, 4, 4));
    BOOST_CHECK_EQUAL(3, getLargestSameMod(3, 5, 4));
    BOOST_CHECK_EQUAL(3, getLargestSameMod(3, 6, 4));
    BOOST_CHECK_EQUAL(7, getLargestSameMod(3, 7, 4));

    BOOST_CHECK_EQUAL(0, getRequired(0, 4));
    BOOST_CHECK_EQUAL(1, getRequired(1, 4));
    BOOST_CHECK_EQUAL(1, getRequired(4, 4));
    BOOST_CHECK_EQUAL(2, getRequired(5, 4));
}

BOOST_AUTO_TEST_CASE(testSolverPassesManyAssignments) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 64, 3, 100);

    GpuHelpedSolver& solver = *(fx.solvers[0]);
    for (int i = 0; i < 32; i++) {
        addClause(*fx.co.hClauses, ~mkLit(2 * i), mkLit(2 * i + 1));
        solver.cancelUntil(0);
        solver.newDecisionLevel();
        solver.uncheckedEnqueue(mkLit(2 * i));
        solver.copyAssigsForGpu(solver.decisionLevel());
    }
    execute(*fx.co.gpuRunner);

    bool foundEmptyClause = false;
    solver.gpuImportClauses(foundEmptyClause);

    BOOST_CHECK_EQUAL(32, solver.stats[nbReported]);
    BOOST_CHECK_EQUAL(1, solver.stats[nbImportedValid]);
    BOOST_CHECK_EQUAL(32, solver.stats[nbImported]);

    // check the clauses have really been learned and can be propagated
    solver.cancelUntil(0);
    for (int i = 0; i < 32; i++) {
        solver.newDecisionLevel();
        solver.uncheckedEnqueue(mkLit(2 * i));
        BOOST_CHECK(l_Undef == solver.value(2 * i + 1));
        solver.propagate();
        BOOST_CHECK(l_True == solver.value(2 * i + 1));
    }
}

BOOST_AUTO_TEST_CASE(testGpuMultiSolver) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 2, 1);

    GpuMultiSolver &msolver = *fx.co.gpuMultiSolver;

    vec<Lit> lits;
    lits.push(mkLit(0));
    msolver.addClause_(lits);
    lits.clear();
    lits.push(~mkLit(0));
    lits.push(mkLit(1));
    msolver.addClause_(lits);
    BOOST_CHECK((l_True == msolver.solve(1)));
}

BOOST_AUTO_TEST_CASE(testSendClauseToGpu) {
    GpuOptions ops;
    setDefaultOptions(ops);
    GpuFixture fx(ops, 3, 1);
    GpuHelpedSolver& solver = *(fx.solvers[0]);

    solver.addClause(~mkLit(0), mkLit(1));
    solver.addClause(~mkLit(0), ~mkLit(1));
    solver.newDecisionLevel();
    solver.uncheckedEnqueue(mkLit(0));

    BOOST_CHECK_EQUAL(0, solver.conflicts);
    BOOST_CHECK_EQUAL(0, solver.stats[propagations]);

    bool b1;
    vec<Lit> learned_clause, selectors;
    bool blocked = false;
    solver.propagateAndMaybeLearnFromConflict(b1, blocked, learned_clause, selectors);
    BOOST_CHECK_EQUAL(1, solver.conflicts);
    BOOST_CHECK_EQUAL(1, solver.stats[propagations]);
    execute(*fx.co.gpuRunner);

    solver.gpuImportClauses(b1);
    // if already present:
    // The gpu clauses don't need to learn the clause ~mkLit(0) that the solver has just found because the assignment mkLit(0) was
    // already know to be not useful (because the gpu clauses already have the clause ~mkLit(0)
    copyToDeviceAsync(*fx.co.hClauses, fx.sp.get(), fx.co.gpuDims);
    BOOST_CHECK_EQUAL(1, fx.co.hClauses->getClauseCount());
    BOOST_CHECK_EQUAL(1, fx.co.hClauses->getClauseLengthSum());
}

BOOST_AUTO_TEST_CASE(testClauseBatch) {
    ClauseBatch clBatch;
    clBatch.addClause(23);
    clBatch.addLit(mkLit(1));
    clBatch.addLit(mkLit(2));

    MinHArr<Lit> minHArr;
    GpuClauseId gpuClauseId;
    BOOST_CHECK((clBatch.popClause(minHArr, gpuClauseId)));
    BOOST_CHECK_EQUAL(23, gpuClauseId);
    BOOST_CHECK_EQUAL(2, minHArr.size());
    BOOST_CHECK_EQUAL(toInt(mkLit(1)), toInt(minHArr[0]));
    BOOST_CHECK_EQUAL(toInt(mkLit(2)), toInt(minHArr[1]));
}


BOOST_AUTO_TEST_SUITE_END()

}
