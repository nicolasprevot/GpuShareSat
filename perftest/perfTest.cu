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
#define BOOST_TEST_MODULE perftest_module
#include <boost/test/unit_test.hpp>
#include "gpu/Helper.cuh"
#include "gpu/Assigs.cuh"
#include "gpu/Clauses.cuh"
#include "gpu/GpuHelpedSolver.h"
#include "gpu/GpuRunner.cuh"
#include "satUtils/SolverTypes.h"
#include "core/Solver.h"
#include "gpuShareLib/Utils.h"
#include <cuda.h>
#include <mtl/Vec.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <memory>
#include "gpu/GpuRunner.cuh"
#include "testUtils/TestHelper.cuh"
#include "gpuShareLib/Utils.h"
#include "gpu/my_make_unique.h"
#include "utils/Utils.h"

namespace Glucose {

int getDiffMicros(timespec begin, timespec end) {
    return (end.tv_sec - begin.tv_sec) * 1000000 + (end.tv_nsec - begin.tv_nsec) / 1000;
}

std::unique_ptr<GpuOptions> getOptions(int clCount, int clMinSize, int clMaxSize) {
    auto ptr = my_make_unique<GpuOptions>();
    ptr -> blockCount = 10;
#ifndef NDEBUG
    ptr -> threadsPerBlock = 150;
#else
    ptr -> threadsPerBlock = 1024;
#endif
    // make sure that we don't reduce the db
    ptr -> gpuFirstReduceDb = 1e9;
    return ptr;
}


class PerfFixture : public GpuFixture {
public:
    int clauseCount;
    int clMinSize;
    int clMaxSize;
    PerfFixture(int clCount = 1000000, int clMinSize = 12, int clMaxSize = 20, int varCount = 500, int solverCount = 1);
};

void maybeSetVariable(double &seed, GpuHelpedSolver &solver, int var) {
    int p = irand(seed, 3);
    if (p == 0 || p == 1) {
        solver.newDecisionLevel();
        solver.uncheckedEnqueue(mkLit(var, p == 1));
    }
}

void resetAllVariables(double &seed, GpuHelpedSolver &solver) {
    solver.cancelUntil(0);
    for (int i = 0; i < solver.nVars(); i++) {
        maybeSetVariable(seed, solver, i);
    }
}

// This has to be set before the gpu starts, so at the beginning of each test
void setDeviceFlags() {
    unsigned int flags;
    cudaGetDeviceFlags(&flags);
    if (flags & cudaDeviceBlockingSync == 0) {
        exitIfError(cudaSetDeviceFlags(cudaDeviceBlockingSync), POSITION);
    }
}

PerfFixture::PerfFixture(int _clauseCount, int _clMinSize, int _clMaxSize, int nVars, int solverCount) :
    clauseCount(_clauseCount),
    clMinSize(_clMinSize),
    clMaxSize(_clMaxSize),
    GpuFixture(*(getOptions(_clauseCount, _clMinSize, _clMaxSize)), nVars, solverCount, 2000) {
    srand(25);
    vec<Lit> lits;
    ContigCopier cc(true);
    cudaStream_t &stream = gpuClauseSharer.sp.get();
    GpuDims gpuDims {10, 256};
    double seed = 0.4;
    for (int cl = 0; cl < clauseCount; cl++) {
        lits.clear();
        int size = irand(seed, clMinSize, clMaxSize);
        for (int l = 0; l < size; l++) {
            lits.push(randomLit(seed, nVars));
        }
        gpuClauseSharer.clauses->addClause(MinHArr<Lit>(lits.size(), &lits[0]), 5);
        // HClauses is designed to copy clauses in small chunks, not a large amount at once
        if (cl % 5000 == 0) {
            copyToDeviceAsync(*gpuClauseSharer.clauses, stream, cc, gpuDims);
            exitIfError(cudaStreamSynchronize(stream), POSITION);
        }
    }
    copyToDeviceAsync(*gpuClauseSharer.clauses, stream, cc, gpuDims);
    exitIfError(cudaStreamSynchronize(stream), POSITION);
}

// print all the wrong clauses
BOOST_AUTO_TEST_CASE(testPrintClauses) {
    setDeviceFlags();
    PerfFixture fx(300000, 10, 11);
    double seed = 0.6;
    resetAllVariables(seed, *(fx.solvers[0]));
    fx.solvers[0]->tryCopyTrailForGpu(fx.solvers[0]->decisionLevel());
    execute(fx.gpuClauseSharer);
    Lit array[MAX_CL_SIZE];
    GpuClauseId gpuClauseId;
    MinHArr<Lit> lits;

    while (fx.gpuClauseSharer.reported->popReportedClause(0, lits, gpuClauseId)) {
        // vec doesn't have a sort method, so let's use an array instead
        for (int j = 0; j < lits.size(); j++) {
            array[j] = lits[j];
        }
        std::sort(array, array + lits.size());
        printf(">> ");
        for (int j = 0; j < lits.size(); j++) {
            PRINTV(array[j]);
        }
        printf("\n");
    }
}

BOOST_AUTO_TEST_CASE(testPerf) {
    setDeviceFlags();
    PerfFixture fx(2000000, 12, 20, 500, 1);

    exitIfLastError(POSITION);
    timespec begin, gpuDone, end;
    long gpuExecTimeMicros = 0;
    long importTimeMicros = 0;
    exitIfLastError(POSITION);
    // having n = 2000 is really to slow if we're in debug
    // But in release, to have a consistent result, we need a big enough
    // value for n
#ifdef NDEBUG
    long n = 2000;
#else
    long n = 15;
#endif

    double seed = 0.6;
    printf("solver count: %d\n", fx.solvers.size());
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < fx.solvers.size(); j++) {
            resetAllVariables(seed, *(fx.solvers[j]));
            fx.solvers[j]->tryCopyTrailForGpu(fx.solvers[j]->decisionLevel());
        }
        clock_gettime(CLOCK_REALTIME, &begin);
        execute(fx.gpuClauseSharer);
        clock_gettime(CLOCK_REALTIME, &gpuDone);
        // This is partly because we can't add more assignments unless we read clauses for existing assignments
        bool a;
        for (int j = 0; j < fx.solvers.size(); j++) fx.solvers[j]->gpuImportClauses(a);
        exitIfLastError(POSITION);
        clock_gettime(CLOCK_REALTIME, &end);
        gpuExecTimeMicros += getDiffMicros(begin, gpuDone);
        importTimeMicros += getDiffMicros(gpuDone, end);
    }

    if (gpuExecTimeMicros + importTimeMicros == 0) {
        printf("no time passed");
    }
    else {
        printf("gpu exec time taken: %ld micros\n", gpuExecTimeMicros);
        printf("import time taken: %ld micros\n", importTimeMicros);
        printf("wrong clause count: %ld\n", fx.gpuClauseSharer.getGlobalStat(gpuReports));
        printf("clause count: %d\n", fx.clauseCount);
        printf("executions per seconds: %ld\n", (n * 1000000)/ (gpuExecTimeMicros + importTimeMicros));
        printf("reads per microsecond: %ld\n", n * fx.clauseCount * (1 + (fx.clMinSize + fx.clMaxSize) / 2) / (gpuExecTimeMicros));
    }
#ifdef NDEBUG
    BOOST_CHECK_EQUAL(19739, fx.gpuClauseSharer.getGlobalStat(gpuReports));
#else
    BOOST_CHECK_EQUAL(143, fx.gpuClauseSharer.getGlobalStat(gpuReports));
#endif
    exitIfLastError(POSITION);
}

BOOST_AUTO_TEST_CASE(testReportedAreValid) {
    setDeviceFlags();
    PerfFixture fx(1000000, 10, 11, 500);
    GpuHelpedSolver &solver = *(fx.solvers[0]);
    exitIfLastError(POSITION);
    bool foundEmptyClause = false;
    int importedValidLastTime = 0;
    int importedLastTime = 0;
    double seed = 0.8;
    resetAllVariables(seed, *(fx.solvers[0]));
    // If the gpu reports some clauses: at least one of them must be valid
    // Because the cpu solver then changes its variables because of this one,
    // the next clauses may not be valid
    while (true) {
        fx.solvers[0]->tryCopyTrailForGpu(fx.solvers[0]->decisionLevel());
        // the first maybExecute will only start the run but not get the results, so execute twice
        execute(fx.gpuClauseSharer);
        CRef conflict = solver.gpuImportClauses(foundEmptyClause);
        int reported = solver.stats[nbImported], importedValid = solver.stats[nbImportedValid];
        printf("%d clauses imported out of which %d valid\n", reported, importedValid);

        vec<Lit> clauseLits;

        // continue as long as we get some clauses
        if (solver.stats[nbImported] == importedLastTime) {
            break;
        }
        importedLastTime = solver.stats[nbImported];
        ASSERT_OP(solver.stats[nbImportedValid], >, importedValidLastTime);
        importedValidLastTime = solver.stats[nbImportedValid];

        // If the solver got a conflict at level n, it's still at level n.
        // We need to cancel it until the previous level because otherwise, it will get the same conflict over and over
        if (conflict != CRef_Undef) {
            if (solver.decisionLevel() == 0) break;
            solver.cancelUntil(solver.decisionLevel() - 1);
        }
    }
    exitIfError(cudaStreamSynchronize(fx.gpuClauseSharer.sp.get()), POSITION);
    exitIfLastError(POSITION);
}

}
