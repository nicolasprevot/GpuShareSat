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
 * CompositionRoot.h
 *
 *  Created on: 14 Jul 2018
 *      Author: nicolas
 */

#ifndef COMPOSITIONROOT_H_
#define COMPOSITIONROOT_H_

#include "utils/Options.h"
#include "GpuUtils.cuh"
#include "CorrespArr.cuh"
#include "Assigs.cuh"
#include "Reported.cuh"
#include "GpuMultiSolver.cuh"
#include "Reporter.cuh"
#include "ContigCopy.cuh"
#include "Clauses.cuh"
#include "GpuRunner.cuh"
#include "GpuHelpedSolver.cuh"
#include "utils/Periodic.h"
#include "my_make_unique.h"
#include "satUtils/InitHelper.h"

#include <memory>

namespace Glucose {

class GpuOptions {
public:
    IntOption blockCount;
    IntOption threadsPerBlock;
    DoubleOption gpuClauseActivityDecay;
    IntOption gpuFirstReduceDb;
    IntOption gpuIncReduceDb;
    IntOption writeClausesPeriodSec;
    IntOption writeStatsPeriodSec;
    IntOption minGpuLatencyMicros;
    BoolOption gpuActOnly;
    GpuHelpedSolverOptions gpuHelpedSolverOptions;
    IntOption solverCount;
    IntOption maxSolverCount;
    IntOption maxMemory;
    BoolOption quickProf;

    GpuOptions();
    int getNumberOfCpuThreads(int verbosity, float mem);
};

class CompositionRoot {
public:
    // The reason for having them public is that they're used by the tests as well,
    // and the tests need to look at these individually
    int varCount;
    GpuDims gpuDims;
    StreamPointer streamPointer;
    CorrespArr<int> clausesCountPerThread;
    std::unique_ptr<HostAssigs> hostAssigs;
    std::unique_ptr<HostClauses> hClauses;
    std::unique_ptr<Reported> reported;
    std::unique_ptr<GpuRunner> gpuRunner;
    vec<GpuHelpedSolver*> solvers;
    std::unique_ptr<GpuMultiSolver> gpuMultiSolver;
    Verbosity verb;

    CompositionRoot(GpuOptions ops, CommonOptions commonOpts, Finisher &finisher, int varCount, int initRepCountPerCategory = 10);
};

} /* namespace Glucose */

#endif /* COMPOSITIONROOT_H_ */
