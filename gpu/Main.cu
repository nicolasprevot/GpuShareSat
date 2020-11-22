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
#include <sys/time.h>
#include <sys/resource.h>
#include <thread>
#include <csignal>

#include "Helper.cuh"
#include "GpuMultiSolver.cuh"
#include "satUtils/Dimacs.h"
#include "CompositionRoot.cuh"
#include "core/Solver.h"
#include "satUtils/InitHelper.h"
#include "utils/System.h"

using namespace Glucose;

Finisher finisher;

void SIGINT_exit(int signum) {
    finisher.cancel();
}


int runGpuSolver(CompositionRoot &compRoot, GpuOptions &gpuOptions, CommonOptions &commonOpts, double memUsedOneSolver) {
    GpuMultiSolver& msolver = *compRoot.gpuMultiSolver;
    msolver.setVerbosity(commonOpts.getVerbosity());
    lbool ret = l_Undef;
    if (commonOpts.doPreprocessing()) {
        ret = msolver.simplify();
    }
    if (ret == l_Undef) {
        // We have an approximation of the memory used for one solver. We don't take into account the memory used for the gpu itself
        // or other things.
        int cpuSolverCount = gpuOptions.getNumberOfCpuThreads(msolver.getVerbosity().global, memUsedOneSolver);
        int warpsPerBlock = compRoot.gpuDims.threadsPerBlock / WARP_SIZE;
        compRoot.hostAssigs->growSolverAssigs(cpuSolverCount, warpsPerBlock, compRoot.gpuDims.blockCount * warpsPerBlock);
        ret = msolver.solve(cpuSolverCount);
    }
    printResult(ret);

    if (msolver.getVerbosity().showModel && ret==l_True) {
        printModel(stdout, msolver.getModel());
    }
    return getReturnCode(ret);
}

int main(int argc, char **argv)
{
    // It can be necessary for debugging if we print a lot
    // 100 Megs
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1048576 * 100);

    printf("c\nc This is glucose-gpu 1.0 --  based on MiniSAT (Many thanks to MiniSAT team)\nc\n");
    signal(SIGINT, SIGINT_exit);
    signal(SIGXCPU,SIGINT_exit);
    try {
        CommonOptions commonOptions;
        GpuOptions gpuOptions;
        setUsageHelp("c USAGE: %s [options] <input-file> <result-output-file>\n\n  where input may be either in plain or gzipped DIMACS.\n");
        parseOptions(argc, argv, true);
        commonOptions.applyTimeAndCpuLim();
        TimePrinter timePrinter("taken total");

        gzFile in = getInputFile(argc, argv);

        DimacsParser parser(in);

        // Note: seems this flag has to be set before a stream is created (rather than on the thread using the stream)
        // The reason for setting it here instead of composition root is that:
        // If set in comp root, it would be set for some unit tests but not other
        // but this flag needs to be set before the device starts to run, so it wouldn't work
        // this flag is good if there are few threads
        // exitIfError(cudaSetDeviceFlags(cudaDeviceBlockingSync), POSITION);
        CompositionRoot compRoot(gpuOptions, finisher, parser.nVars());

        // Note: cuda uses a gigantic amount (gigabytes) of virtual memory that is almost never used, to get the whole physical memory into virtual
        // memory. We'd like to not count that in the memory limits. But not easy to get just this one

        // Unfortunately, memUsed here includes some memory that is really used, but it's not easy to tell what is
        commonOptions.applyMemLim(memUsed());
        double memUsedBeforeClauses = memUsed();
        parser.fillClauses(*compRoot.gpuMultiSolver);
        double memUsedAfterClauses = memUsed();
        gzclose(in);

        // We only look at differentials of memory to get the memory per solver
        return runGpuSolver(compRoot, gpuOptions, commonOptions, memUsedAfterClauses - memUsedBeforeClauses
                + compRoot.gpuMultiSolver->getMemUsedCreateOneSolver());
    } catch (OutOfMemoryException&){
        printf("c ===================================================================================================\n");
        printf("INDETERMINATE. OutOfMemoryException\n");
        exit(0);
    }
}
