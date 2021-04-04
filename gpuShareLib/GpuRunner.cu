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
#include "Helper.cuh"
#include "GpuRunner.cuh"
#include "Clauses.cuh"
#include "Reported.cuh"
#include "Assigs.cuh"
#include "Reporter.cuh"
#include "gpuShareLib/Utils.h"
#include "CorrespArr.cuh"
#include "my_make_unique.h"
#include <thread>         // std::this_thread::sleep_for
#include "GpuClauseSharer.h"

// #define PRINTCN_ALOT 1

namespace GpuShare {
struct ReportComputer {
    // if all literals seen so far are false in the current clause
    Vals allFalse;
    // all are false, exect for one which is undefined
    Vals justOneUndefined;

    __device__ void init(Vals startVal) {
        justOneUndefined = 0;
        allFalse = startVal;
    }

    __device__ bool nothingToReport() {
        return !(allFalse | justOneUndefined);
    }

    __device__ void update(Vals canBeFalse, Vals canBeUndef) {
        justOneUndefined = (allFalse & canBeUndef) | (justOneUndefined & canBeFalse);
        allFalse = allFalse & canBeFalse;
    }

    __device__ Vals getToReport() {
        return allFalse | justOneUndefined;
    }
};

__global__ void init(DValsPerId<VarUpdate> varUpdates, DArr<DOneSolverAssigs> dOneSolverAssigs, DAssigAggregates aggregates, DReporter<ReportedClause> dReporter, DValsPerId<AggCorresp> aggCorresps) {
    dReporter.clear();
    dUpdateAssigs(varUpdates, dOneSolverAssigs, aggCorresps, aggregates); 
}

__global__ void initClauses(DClauseUpdates dClauseUpdates, DClauses dClauses) {
    updateClauses(dClauseUpdates, dClauses);
}

__device__ void dCheckOneClauseOneSolver(DOneSolverAssigs dOneSolverAssigs, DAssigAggregates dAssigAggregates,
        Lit* startLitPt, Lit* endLitPt, DReporter<ReportedClause> dReporter, int solverId, GpuCref gpuCref) {
    Lit* litPt = startLitPt;
    ReportComputer reportComputer;
    reportComputer.init(dOneSolverAssigs.startVals);
    while (litPt < endLitPt) {
        Lit lit = *litPt;
        Vals va = dVar(lit);
        Vals isFalse = dOneSolverAssigs.multiLBools[va].isTrue;
        if (!dSign(lit)) {
            isFalse = ~ isFalse;
        }
        Vals def = dOneSolverAssigs.multiLBools[va].isDef;
        reportComputer.update(isFalse & def, ~def);
        if (reportComputer.nothingToReport()) {
            return;
        }
        litPt += WARP_SIZE;
    }
    ASSERT_OP_C(gpuCref.clSize, >=, 1);
    dReporter.report(ReportedClause {reportComputer.getToReport(), solverId, gpuCref}, getThreadId());
}

// note: this method doesn't unset the first bit, unlike the cpu one
__device__ void dGetFirstBitPosFast(Vals &val, int &pos) {
    if ((val & 0xFFFF) == 0) {
        val = val >> 16;
        pos += 16;
    }
    if ((val & 0xFF) == 0) {
        val = val >> 8;
        pos += 8;
    }
    if ((val & 0xF) == 0) {
        val = val >> 4;
        pos += 4;
    }
    if ((val & 0x3) == 0) {
        val = val >> 2;
        pos += 2;
    }
    if ((val & 0x1) == 0) {
        val = val >> 1;
        pos += 1;
    }
    assert(val & 1);
}

__device__ void dCheckOneClauseAllSolvers(DArr<DOneSolverAssigs> dOneSolverAssigs, DAssigAggregates dAssigAggregates,
        Lit* startLitPt, Lit*endLitPt, DReporter<ReportedClause> dReporter, GpuCref gpuCref, Vals bits, long &clauseTestsOnAssigs) {
    int pos = 0;
    while (true) {
        if (bits == 0) {
            return;
        }
        dGetFirstBitPosFast(bits, pos);
        int solver = dAssigAggregates.getSolver(pos);
        dCheckOneClauseOneSolver(dOneSolverAssigs[solver], dAssigAggregates, startLitPt, endLitPt, dReporter, solver, gpuCref);
        int newPos = dAssigAggregates.getEndBitPos(solver);
        bits = bits >> (newPos - pos);
        pos = newPos;
        clauseTestsOnAssigs++;
    }
}

// This method is performance critical. So it's dealing with pointers directly, which isn't super safe
__global__ void dFindClauses(DArr<DOneSolverAssigs> dOneSolverAssigs, DAssigAggregates dAssigAggregates,
        DClauses dClauses, DReporter<ReportedClause> dreporter, DArr<long> oneSolverCheckArr) {
    int clSize, clIdStart, clIdEnd;
    int threadId = getThreadId();
    ReportComputer reportComputer;
    dClauses.getClsForThread(threadId, clSize, clIdStart, clIdEnd);
    ASSERT_OP_C(clSize, >=, 1);
    for (int clId = clIdStart; clId < clIdEnd; clId += WARP_SIZE) {
        Lit *startLitPt = dClauses.getStartAddrForClause(clSize, clId);
        Lit *litPt = startLitPt;
        Lit *endLitPt = litPt + WARP_SIZE * clSize;

        reportComputer.init(dAssigAggregates.startVals);
        GpuCref gpuCref {clSize, clId};
        while (litPt < endLitPt) {
#ifndef NDEBUG
            dClauses.assertInSize(clSize, litPt);
#endif
            Lit lit = *litPt;

            Vals va = dVar(lit);
            ASSERT_OP_MSG_C(va, <, dAssigAggregates.multiAggs.size(), PRINTCN(lit); PRINTCN(clId); PRINTCN(clSize); PRINTCN(dClauses.getClCount(clSize)));
            MultiAgg &multiAgg = dAssigAggregates.multiAggs[va];
            assert((~ (multiAgg.canBeTrue | multiAgg.canBeFalse | multiAgg.canBeUndef)) == 0);
            Vals val;
            if (dSign(lit)) {
                val = multiAgg.canBeTrue;
            }
            else {
                val = multiAgg.canBeFalse;
            }
            reportComputer.update(val, multiAgg.canBeUndef);
            if (reportComputer.nothingToReport()) {
                // gotos are bad, but this code is performance critical so it's worth it
                goto next;
            }
            litPt += WARP_SIZE;
        }
        dCheckOneClauseAllSolvers(dOneSolverAssigs, dAssigAggregates,
            startLitPt, endLitPt, dreporter, gpuCref, reportComputer.getToReport(), oneSolverCheckArr[threadId]);
next: ;
    }
}

GpuRunner::GpuRunner(HostClauses &_hostClauses, HostAssigs &_hostAssigs, Reported &_reported, GpuDims gpuDimsGuideline, bool _quickProf,
        int _countPerCategory, cudaStream_t &_stream, std::vector<unsigned long> &_globalStats, const Logger &logger) :
    warpsPerBlock(gpuDimsGuideline.threadsPerBlock / WARP_SIZE),
    blockCount(gpuDimsGuideline.blockCount),
    hasRunOutOfGpuMemoryOnce(false),
    lastInAssigIdsPerSolver(1),
    clauseTestsOnAssigs(false, false, logger),
    quickProf(_quickProf),
    hostAssigs(_hostAssigs),
    hostClauses(_hostClauses),
    reported(_reported),
    categoryCount(gpuDimsGuideline.blockCount),
    countPerCategory(_countPerCategory), 
    cpuToGpuContigCopier(logger, true),
    gpuToCpuContigCopier(logger, true),
    stream(_stream),
    globalStats(_globalStats) {

}

void GpuRunner::prepareOneSolverChecksAsync(int threadCount, cudaStream_t &stream) {
    int oldSize = clauseTestsOnAssigs.size();
    // gpuThreadCount can change with every run
    if (oldSize < threadCount) {
        clauseTestsOnAssigs.resize(threadCount, false);
        for (int i = oldSize; i < threadCount; i++) {
            clauseTestsOnAssigs[i] = 0;
        }
        clauseTestsOnAssigs.copyAsync(cudaMemcpyHostToDevice, stream, oldSize, threadCount);
    }
}

void GpuRunner::wholeRun(bool canStart) {
    // make sure that we've at least finished the previous copy from cpu to gpu
    exitIfError(cudaEventSynchronize(cpuToGpuCopyDone.get()), POSITION);
    // The gpu is currently processing some assignments. Let's start preparing the next assignments once findClauses is done. At that point, the gpu will still
    // have to set all assigs to last and to copy things back to the cpu
    if (prevReporter) {
        cudaEventSynchronize(afterFindClauses.get());
        if (quickProf) {
            float ms;
            exitIfError(cudaEventElapsedTime(&ms, beforeFindClauses.get(), afterFindClauses.get()), POSITION);
            globalStats[timeSpentTestingClauses] += ms * 1000;
        }
    }
    int nextInAssigIdsPerSolver = -1;
    std::unique_ptr<Reporter<ReportedClause>> nextReporter;
    bool startingNew = false;
    bool notEnoughGpuMemory = false;
    if (canStart) {
        nextInAssigIdsPerSolver = (lastInAssigIdsPerSolver + 1) % 2;
        startGpuRunAsync(stream, assigIdsPerSolver[nextInAssigIdsPerSolver], nextReporter, startingNew, notEnoughGpuMemory);
    }
    if (prevReporter) {
        gatherGpuRunResults(assigIdsPerSolver[lastInAssigIdsPerSolver], *prevReporter);
    }
    if (startingNew) {
        scheduleGpuToCpuCopyAsync(stream);
        lastInAssigIdsPerSolver = nextInAssigIdsPerSolver;
        prevReporter.swap(nextReporter);
    } else {
        prevReporter.reset();
    }
    if (notEnoughGpuMemory) {
        hostClauses.reduceDb(stream);
        hasRunOutOfGpuMemoryOnce = true;
    }
}

/*
struct InitParams {
    DArr<DArr<VarUpdate>> varUpdates;
    DArr<DOneSolverAssigs> dOneSolverAssigs;
    DAssigAggregates aggregates;
    DReporter<ReportedClause> dReporter;
    DClauseUpdates dClauseUpdates;
    DClauses dClauses;
}

*/
void GpuRunner::startGpuRunAsync(cudaStream_t &stream, std::vector<AssigIdsPerSolver> &assigIdsPerSolver, std::unique_ptr<Reporter<ReportedClause>> &reporter, bool &started, bool &notEnoughGpuMemory) {
#ifdef PRINTCN_ALOT
    printf("startGpuRunAsync\n");
#endif

    // note that there could still be some updates from the previous run that still need to be read from the gpu to set all vars to last
    // it is fine to clear here though, since the copy will be enqueued on the same stream as the one where all vars will be set to last
    // so it will happen after
    cpuToGpuContigCopier.clear(false);

    ClUpdateSet clUpdateSet = hostClauses.getUpdatesForDevice(stream, cpuToGpuContigCopier);
    // globalStats[gpuClauses] at this point includes clauses that are about to be copied to the device
    if (globalStats[gpuClauses] == 0) {
        started = false;
        notEnoughGpuMemory = false;
        return;
    }
    RunInfo runInfo = hostClauses.makeRunInfo(stream, cpuToGpuContigCopier);

    if (!runInfo.succeeded()) {
        // Failed to allocate the memory 
        // it's fine not to call initClauses since the next thing we'll do is reduceDb which will
        // sync device and host anyway
        started = false;
        notEnoughGpuMemory = true;
        return;
    }

    TimeGauge tg(globalStats[timeSpentFillingAssigs], quickProf);
    AssigsAndUpdates assigsAndUpdates = hostAssigs.fillAssigsAsync(cpuToGpuContigCopier, assigIdsPerSolver, stream);
    tg.complete();

    if (!cpuToGpuContigCopier.tryCopyAsync(cudaMemcpyHostToDevice, stream)) {
        THROW();
    }
    exitIfError(cudaEventRecord(cpuToGpuCopyDone.get(), stream), POSITION);

    gpuToCpuContigCopier.clear(false);
    reporter = my_make_unique<Reporter<ReportedClause>>(gpuToCpuContigCopier, stream, countPerCategory, categoryCount);
    auto dReporter = reporter->getDReporter();
    DClauses dClauses = runInfo.getDClauses();

    ASSERT_OP_C(warpsPerBlock, >, 0);

    runGpuAdjustingDims(warpsPerBlock, warpsPerBlock * blockCount, [&] (int blockCount, int threadsPerBlock) {
        init<<<blockCount, threadsPerBlock, 0, stream>>>(assigsAndUpdates.dAssigUpdates.get(), assigsAndUpdates.assigSet.dSolverAssigs.getDArr(), assigsAndUpdates.assigSet.dAssigAggregates, dReporter, assigsAndUpdates.assigSet.aggCorresps.get());
    });
    exitIfError(cudaGetLastError(), POSITION);

    runGpuAdjustingDims(warpsPerBlock, warpsPerBlock * blockCount, [&] (int blockCount, int threadsPerBlock) {
        initClauses<<<blockCount, threadsPerBlock, 0, stream>>>(clUpdateSet.getDClauseUpdates(), dClauses);
    });

    prepareOneSolverChecksAsync(runInfo.warpCount * WARP_SIZE, stream);
    if (quickProf) exitIfError(cudaEventRecord(beforeFindClauses.get(), stream), POSITION);

    // Only this run uses runInfo.warpCount for the dimensions
    runGpuAdjustingDims(warpsPerBlock, runInfo.warpCount, [&] (int blockCount, int threadsPerBlock) {
        dFindClauses<<<blockCount, threadsPerBlock, 0, stream>>>(assigsAndUpdates.assigSet.dSolverAssigs.getDArr(),
            assigsAndUpdates.assigSet.dAssigAggregates, dClauses, dReporter, clauseTestsOnAssigs.getDArr());
    });
    exitIfError(cudaEventRecord(afterFindClauses.get(), stream), POSITION);
    setAllAssigsToLastAsync(warpsPerBlock, warpsPerBlock * blockCount, assigsAndUpdates, stream);
    started = true;
    notEnoughGpuMemory = false;
}

void GpuRunner::scheduleGpuToCpuCopyAsync(cudaStream_t &stream) {
    exitIfFalse(gpuToCpuContigCopier.tryCopyAsync(cudaMemcpyDeviceToHost, stream), POSITION);
    exitIfError(cudaEventRecord(gpuToCpuCopyDone.get(), stream), POSITION);
}

int getTotalAssigCount(std::vector<AssigIdsPerSolver> &assigIdsPerSolver) {
    int res = 0;
    for (int i = 0; i < assigIdsPerSolver.size(); i++) {
        res += assigIdsPerSolver[i].assigCount;
    }
    return res;
}

void GpuRunner::gatherGpuRunResults(std::vector<AssigIdsPerSolver> &assigIdsPerSolver, Reporter<ReportedClause> &reporter) {
    globalStats[gpuRuns]++;
    exitIfError(cudaEventSynchronize(gpuToCpuCopyDone.get()), POSITION);
    if (reporter.getCopiedToHost(reportedCls)) {
        countPerCategory *= 2;
    }

    int assigsCount = getTotalAssigCount(assigIdsPerSolver);
    int clCount = globalStats[gpuClauses];
    globalStats[totalAssigClauseTested] += clCount * assigsCount;
    globalStats[clauseTestsOnGroups] += clCount;
    globalStats[gpuReports] += reportedCls.size();
#if PRINTCN_ALOT == 1
    printf("filling reported with %d assigs and %d clauses\n", assigsCount, reportedCls.size());
#endif
    for (int i = 0; i < reportedCls.size(); i++) {
        ASSERT_OP_C(reportedCls[i].gpuCref.clSize, >=, 1);
        hostClauses.bumpClauseActivity(reportedCls[i].gpuCref);
    }
    {
        TimeGauge tg(globalStats[timeSpentFillingReported], quickProf);
        reported.fill(assigIdsPerSolver, reportedCls);
    }
}

long GpuRunner::getClauseTestsOnAssigs() {
    clauseTestsOnAssigs.copyAsync(cudaMemcpyDeviceToHost, stream);
    exitIfError(cudaStreamSynchronize(stream), POSITION);
    return getSum(clauseTestsOnAssigs);
}

}
