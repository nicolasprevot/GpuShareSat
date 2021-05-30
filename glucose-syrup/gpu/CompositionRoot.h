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

#include <iostream>

#include "utils/Options.h"
#include "GpuHelpedSolver.h"
#include "Periodic.h"
#include "satUtils/InitHelper.h"
#include "GpuMultiSolver.h"
#include "gpuShareLib/GpuClauseSharer.h"

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
    GpuHelpedSolverOptions gpuHelpedSolverOptions;
    IntOption solverCount;
    IntOption maxSolverCount;
    IntOption maxMemory;
    BoolOption quickProf;

    GpuOptions();
    int getNumberOfCpuThreads(int verbosity, float mem);

    GpuShare::GpuClauseSharerOptions toGpuClauseSharerOptions(int verbosity, int initRepCountPerCategory = 10);
};


struct SyncedPrinter {
    std::mutex lock;

    void operator()(const std::string &str) {
        std::lock_guard<std::mutex> guard(lock);
        std::cout << str << std::endl;
    }
};

class CompositionRoot {
public:
    // The reason for having them public is that they're used by the tests as well,
    // and the tests need to look at these individually
    int varCount;
    std::unique_ptr<GpuShare::GpuClauseSharer> gpuClauseSharer;
    std::unique_ptr<GpuMultiSolver> gpuMultiSolver;
    SyncedPrinter syncedPrinter;
    Verbosity verb;
    GpuShare::Logger logger;

    CompositionRoot(GpuOptions ops, CommonOptions commonOpts, Finisher &finisher, int varCount);
};

} /* namespace Glucose */

#endif /* COMPOSITIONROOT_H_ */
