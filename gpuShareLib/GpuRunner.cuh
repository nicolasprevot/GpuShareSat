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
#ifndef GpuRunner_h
#define GpuRunner_h

#include "BaseTypes.cuh"
#include "Reporter.cuh"
#include "GpuUtils.cuh"
#include <memory>
#include "Profiler.h"
#include <vector>

namespace GpuShare {

class HostAssigs;
class HostClauses;
class ClauseActivity;
class DAssigs;
class Reported;
class DOneSolverAssigs;
class DAssigAggregates;
class AssigsAndUpdates;

// This class deals with actually running the GPU and checking clauses against assignments
class GpuRunner {
private:
    ContigCopier cpuToGpuContigCopier;
    ContigCopier gpuToCpuContigCopier;

    int warpsPerBlock;
    int blockCount;
    bool hasRunOutOfGpuMemoryOnce;
    EventPointer beforeFindClauses;
    EventPointer afterFindClauses;

    EventPointer gpuToCpuCopyDone;
    EventPointer cpuToGpuCopyDone;

    std::vector<ReportedClause> reportedCls;
    int lastInAssigIdsPerSolver;
    std::vector<AssigIdsPerSolver> assigIdsPerSolver[2];
    std::unique_ptr<Reporter<ReportedClause>> prevReporter;

    CorrespArr<long> clauseTestsOnAssigs;
    void prepareOneSolverChecksAsync(int threadCount, cudaStream_t &tream);
    // if we do some simple profiling
    bool quickProf;
    int categoryCount;
    int countPerCategory;
    int minLatencyMicros;
    HostAssigs &hostAssigs;
    HostClauses &hostClauses;
    Reported &reported;

    float timeToWaitSec;
    cudaStream_t &stream;
    std::vector<unsigned long> &globalStats;

    void startGpuRunAsync(cudaStream_t &stream, std::vector<AssigIdsPerSolver> &assigIdsPerSolver, std::unique_ptr<Reporter<ReportedClause>> &reporter, bool &started, bool &notEnoughGpuMemory);
    void scheduleGpuToCpuCopyAsync(cudaStream_t &stream);
    void gatherGpuRunResults(std::vector<AssigIdsPerSolver> &assigIdsPerSolver, Reporter<ReportedClause> &reporter);

public:
    GpuRunner(HostClauses &_hostClauses, HostAssigs &_hostAssigs, Reported &_reported, GpuDims gpuDimsGuideline, bool _quickProf, int _countPerCategory, cudaStream_t &stream, std::vector<unsigned long> &globalStats);

    void wholeRun(bool canStart);
    bool getHasRunOutOfGpuMemoryOnce() { return hasRunOutOfGpuMemoryOnce; }
    long getClauseTestsOnAssigs();
};

}

#endif
