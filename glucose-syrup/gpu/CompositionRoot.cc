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
 * CompositionRoot.cc
 *
 *  Created on: 14 Jul 2018
 *      Author: nicolas
 */

#include "CompositionRoot.h"
#include "GpuHelpedSolver.h"
#include "../utils/System.h"
#include "../utils/Options.h"
#include "gpuShareLib/GpuClauseSharer.h"
#include "gpuShareLib/my_make_unique.h"
#include <thread>
#include <stdio.h>

namespace Glucose {

GpuOptions::GpuOptions():
    blockCount("GPU", "block-count", "Guideline for number of gpu blocks. -1 for auto adjust", -1, IntRange(-1, INT32_MAX)),
    threadsPerBlock("GPU", "threads-per-block", "Number of threads per block. -1 for default", -1),
    gpuClauseActivityDecay("GPU", "gpu-cl-decay", "activity decay for gpu clauses, -1 for default", -1),
    gpuFirstReduceDb("GPU", "gpu-first-reduce-db", "Wait until there are that many clauses on the gpu before the first reduce DB",
            800000, IntRange(0, INT32_MAX)),
    gpuIncReduceDb("GPU", "gpu-inc-reduce-db", "Increase the reduce db count by this much each time we reduce",
            30000, IntRange(0, INT32_MAX)),
    writeClausesPeriodSec("GPU", "write-clauses-period-sec", "Each time this interval is passed, write all clauses", -1),
    writeStatsPeriodSec("GPU", "write-stats-period-sec", "Each time this interval is passed, write all stats. Set to -1 to never write them", 5),
    minGpuLatencyMicros("GPU", "min-gpu-latency-micros", "If a gpu run takes less than this amount of time, wait for the remaining time, -1 for default", -1),
    solverCount("GPU", "thread-count", "Number of core CPU threads for syrup (0 for automatic)", 0),
    maxSolverCount("GPU", "max-thread-count", "Maximum number of core CPU threads to ask for (when thread-count=0). 0 (default) stands for concurrency - 1", 0),
    maxMemory("GPU", "max-memory", "Maximum memory to use (in Mb, 0 for no software limit)", 8000),
    quickProf("GPU", "quick-prof", "if we do some quick and simple profiling. It still makes things slower, but not by too much. It is meant to be used in releases", false)
{

}

GpuShare::GpuClauseSharerOptions GpuOptions::toGpuClauseSharerOptions(int verbosity, int initRepCountPerCategory) {
    GpuShare::GpuClauseSharerOptions csOpts;
    csOpts.gpuBlockCountGuideline = blockCount;
    csOpts.gpuThreadsPerBlockGuideline = threadsPerBlock;
    csOpts.minGpuLatencyMicros = minGpuLatencyMicros;
    csOpts.verbosity = verbosity;
    csOpts.clauseActivityDecay = gpuClauseActivityDecay;
    csOpts.quickProf = quickProf;
    csOpts.initReportCountPerCategory = initRepCountPerCategory;
    csOpts.maxPageLockedMemory = -1; // set to default
    return csOpts;
}

int GpuOptions::getNumberOfCpuThreads(int verbosity, float mem) {
    if (solverCount != 0) return solverCount;
    if (maxSolverCount == 0) {
        int hardwareConcurrency = std::thread::hardware_concurrency();
        if (hardwareConcurrency == 0) {
            hardwareConcurrency = 8;
        }
        if (hardwareConcurrency >= 4) {
            // There's another thread to just run the GPU
            maxSolverCount =  hardwareConcurrency - 1;
        } else {
            maxSolverCount =  hardwareConcurrency;
        }
    }
    if(verbosity >= 1)
        printf("c |  Automatic Adjustement of the number of solvers. MaxMemory=%5d, MaxCores=%3d.                       |\n",
                (int32_t) maxMemory, (int32_t) maxSolverCount);
    int tmpnbsolvers = maxMemory * 4 / 10 / mem;
    if(tmpnbsolvers > maxSolverCount) tmpnbsolvers = maxSolverCount;
    if(tmpnbsolvers < 1) tmpnbsolvers = 1;
    if(verbosity >= 1)
        printf("c |  One Solver is taking %.2fMb... Let's take %d solvers for this run (max 40%% of the maxMemory).       |\n", mem, tmpnbsolvers);
    return tmpnbsolvers;
}

// We don't know yet the number of solvers in this method
// Reason is that we need to look at memory usage of one solver to decide how many solvers to use
// And we need to already have one cpu solver for that 
CompositionRoot::CompositionRoot(GpuOptions opts, CommonOptions commonOpts, Finisher &finisher, int varCount) :
    varCount(varCount),
    verb(commonOpts.getVerbosity()),
    logger {verb.global, [&](const std::string &str) {syncedPrinter(str); }}
{
    GpuShare::GpuClauseSharerOptions csOpts = opts.toGpuClauseSharerOptions(verb.global);
    verb.writeStatsPeriodSec = (verb.global > 0) ? opts.writeStatsPeriodSec : -1;


    gpuClauseSharer = std::unique_ptr<GpuShare::GpuClauseSharer>(GpuShare::makeGpuClauseSharerPtr(csOpts, logger.logFunc));
    //  It matters that this is called after creating the gpu clause sharer, so initMemUsed has the gigantic amount allocated by cuda
    double initMemUsed = memUsed();
    gpuClauseSharer->setVarCount(varCount);

    gpuMultiSolver = my_make_unique<GpuMultiSolver>(finisher, *gpuClauseSharer,
                std::function<GpuHelpedSolver* (int)> ([&](int cpuThreadId) {
                    return new GpuHelpedSolver(finisher, cpuThreadId, opts.gpuHelpedSolverOptions.toParams(), *gpuClauseSharer, opts.quickProf, logger);
                }), varCount, opts.writeClausesPeriodSec, verb, initMemUsed, (double) opts.maxMemory, opts.gpuFirstReduceDb, opts.gpuIncReduceDb, logger);
}

} /* namespace Glucose */
