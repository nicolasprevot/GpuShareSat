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
#include "GpuMultiSolver.h"
#include "GpuHelpedSolver.h"
#include "utils/System.h"
#include "utils/Utils.h"
#include "satUtils/Dimacs.h"
#include <thread> 
#include "utils/Periodic.h"
#include "my_make_unique.h"

namespace Glucose {

// There should only be one instance of this class.
// This class synchronizes the work of several GpuHelpedSolver
GpuMultiSolver::GpuMultiSolver(Finisher &_finisher, GpuClauseSharer &_gpuClauseSharer, 
        std::function<GpuHelpedSolver* (int threadId) > _solverFactory, int varCount, int writeClausesPeriodSec,
        Verbosity _verb, double _initMemUsed, double _maxMemory, int _gpuReduceDbPeriod, int _gpuReduceDbPeriodInc):
                gpuClauseSharer(_gpuClauseSharer),
                solverFactory(_solverFactory),
                verb(_verb),
                gpuReduceDbPeriod(_gpuReduceDbPeriod),
                gpuReduceDbPeriodInc(_gpuReduceDbPeriodInc),
                initMemUsed(_initMemUsed),
                maxMemory(_maxMemory),
                hasTriedToLowerCpuMemoryUsage(false),
                finisher(_finisher) {
    periodicRunner = my_make_unique<PeriodicRunner>(realTimeSecSinceStart()); 
    periodicRunner->add(verb.writeStatsPeriodSec, std::function<void ()> ([&] () { 
        printStats();
    }));
    periodicRunner->add(writeClausesPeriodSec, std::function<void ()> ([&] () {
        writeClausesInCnf();
    }));
    
    
    float memBefore = memUsed();
    helpedSolvers.growTo(1);
    helpedSolvers[0] = solverFactory(0);
    for (int v = 0; v < varCount; v++) {
        helpedSolvers[0]->newVar();
    }
    memUsedCreateOneSolver =  memUsed() - memBefore;
}

void GpuMultiSolver::addClause_(vec<Lit>& lits) {
    assert(helpedSolvers.size() == 1);
    helpedSolvers[0]->addClause_(lits);
}

void GpuMultiSolver::addClause(const vec<Lit>& lits) {
    assert(helpedSolvers.size() == 1);
    helpedSolvers[0]->addClause(lits);
}

void launchSolver(std::mutex &mutex, GpuHelpedSolver*& solver) {
    solver->solve();
    lbool status = solver->getStatus();
    if (status == l_Undef) {
        GpuHelpedSolver *copy = solver;
        {
            std::lock_guard<std::mutex> lock(mutex);
            // solvers which didn't find an answer are no longer useful, destroy them to free memory
            solver = NULL;
        }
        delete copy;
    }
    printf("c A thread is exiting\n");
}

lbool GpuMultiSolver::simplify() {
    int ret2 = helpedSolvers[0]->simplify();
    if(ret2) helpedSolvers[0]->eliminate(true);
    if (!helpedSolvers[0]->okay()) {
        printf("Solved by unit propagation\n");
        return l_False;
    }
    return l_Undef;
}

lbool GpuMultiSolver::solve(int _cpuThreadCount) {
    cpuSolverCount = _cpuThreadCount;
    helpedSolvers.growTo(cpuSolverCount);
    finisher.stopAllThreadsAfterId = _cpuThreadCount;
    for (int i = 1; i < cpuSolverCount; i++) {
        helpedSolvers[i] = new GpuHelpedSolver(*helpedSolvers[0], i);
    }
    configure();

    if (verb.global > 0) {
        printf("c |  all clones generated. Memory = %6.2fMb.                                                             |\n", memUsed());
        printf("c ========================================================================================================|\n");
    }
    gpuClauseSharer.setCpuSolverCount(_cpuThreadCount);
    vec<std::thread> threads;
    long maxApprMemAllocated = -1;
    // Launching all solvers
    threads.growTo(helpedSolvers.size());
    for(int i = 0; i < helpedSolvers.size(); i++) {
        threads[i] = std::thread(launchSolver, std::ref(solversMutex), std::ref(helpedSolvers[i]));
    }
    if (verb.global > 0) SYNCED_OUT(printf("All solvers launched\n"));

    while (!finisher.stopAllThreads) {
        periodicRunner->maybeRun(realTimeSecSinceStart());
        gpuClauseSharer.gpuRun();
        if (gpuClauseSharer.getAddedClauseCount() - gpuClauseSharer.getAddedClauseCountAtLastReduceDb() >= gpuReduceDbPeriod) {
            gpuClauseSharer.reduceDb();
            if (!gpuClauseSharer.hasRunOutOfGpuMemoryOnce()) {
                gpuReduceDbPeriod += gpuReduceDbPeriodInc;
            }
        }

        double cpuMemUsed = actualCpuMemUsed();
        if (!hasTriedToLowerCpuMemoryUsage && cpuMemUsed > 0.9 * maxMemory) {
            // We're not very strict about memory usage on the cpu. Reason
            // is that if we use more than physical memory, it will swap
            // It's very different for gpu memory usage where there's no swap
            // so it crashes if we go over the limit
            if (verb.global > 0) SYNCED_OUT(printf("c There is %lf megabytes of memory used on cpu which is high, the limit is %lf, going to try reducing memory usage\n", cpuMemUsed, maxMemory));
            // All the clauses on the GPU are also on the CPU, so limit the growth of their number
            gpuReduceDbPeriodInc = 0;
            std::lock_guard<std::mutex> lock(solversMutex);
            for (int i = 0; i < helpedSolvers.size(); i++) {
                if (helpedSolvers[i] != NULL) helpedSolvers[i]->tryReduceCpuMemoryUsage();
            }
            hasTriedToLowerCpuMemoryUsage = true;
        }
        // The tryReduceCpuMemoryUsage isn't always very effective, it doesn't limit the growth of permanently learned clauses
        // So we need another mechanism to kill threads if the memory usage is too high. The difficulty is that with the current 
        // way we measure memory usage, if we delete some pointers, it doesn't lead to a decrease of the measured memory usage
        // So instead, compute an approximation of the memory allocated by all the threads at the time when memory usage was too high, 
        // called maxApprMemAllocated
        // If this approximation then becomes higher than that, kill threads.
        if (maxApprMemAllocated == -1 && cpuMemUsed > maxMemory) {
            maxApprMemAllocated = 0;
            for (int i = 0; i < helpedSolvers.size(); i++) {
                maxApprMemAllocated += helpedSolvers[i]->getApproximateMemAllocated();
            }
            if (verb.global > 0) SYNCED_OUT(printf("c There is %lf megabytes of memory used on the cpu when the limit is %lf. Going to kill threads to reduce memory usage\n", cpuMemUsed, maxMemory));
            if (verb.global > 0) SYNCED_OUT(printf("c The approximate memory allocated is %lf\n", maxApprMemAllocated / 1.0e6));
            finisher.stopAllThreadsAfterId = std::max(helpedSolvers.size() - 1, 1);
        }
        if (maxApprMemAllocated > 0)
        for (int i = 0; i < finisher.stopAllThreadsAfterId; i++) {
            long apprMemAllocated = 0;
            for (int i = 0; i < finisher.stopAllThreadsAfterId; i++) {
                std::lock_guard<std::mutex> lock(solversMutex);
                if (helpedSolvers[i] != NULL) {
                    apprMemAllocated += helpedSolvers[i]->getApproximateMemAllocated();
                    if (apprMemAllocated > maxApprMemAllocated && i > 0) {
                        finisher.stopAllThreadsAfterId = i;
                    }
                }
            }
        }
    }
    if (verb.global > 0) {
        SYNCED_OUT(printf("c printing final stats\n"));
        printStats();
    }

    for(int i = 0; i < threads.size(); i++) { // Wait for all threads to finish
        threads[i].join();
    }
    int whoFoundAnAnswer = finisher.oneThreadIdWhoFoundAnAnswer;
    if (whoFoundAnAnswer == -1) return l_Undef;
    return helpedSolvers[whoFoundAnAnswer]->getStatus();
}

vec<lbool>& GpuMultiSolver::getModel() {
    return helpedSolvers[finisher.oneThreadIdWhoFoundAnAnswer]->model;
}

void GpuMultiSolver::printStatSum(const char* name, int stat) {
    printf("c %s: %ld\n", name, getStatSum(stat));
}

void GpuMultiSolver::printStats() {
    SyncOut so;
    static int nbprinted = 1;

    size_t freeGpuMem;
    size_t totalGpuMem;
    long apprMemAllocated = 0;
    gpuClauseSharer.getGpuMemInfo(freeGpuMem, totalGpuMem);
    {
        JStats jstats;
        writeJsonString("type", "periodicStats");
        writeAsJson("cpuTime", cpuTimeSec());
        writeAsJson("realTime", realTimeSecSinceStart());
        {
            writeJsonField("solverStats");
            {
                JArr jarr;
                std::lock_guard<std::mutex> lock(solversMutex);
                for (int i = 0; i < helpedSolvers.size(); i++) {
                    if (helpedSolvers[i] != NULL) {
                        helpedSolvers[i]->printStats();
                        apprMemAllocated += helpedSolvers[i]->getApproximateMemAllocated();
                    }
                }
            }
            writeJsonField("globalStats");
            {
                JObj jo;
                writeAsJson("cpuMemUsed_megabytes", actualCpuMemUsed());
                writeAsJson("gpuMemUsed_megabytes", (float) (totalGpuMem - freeGpuMem) / 1.0e6);
#ifdef KEEP_IMPL_COUNT
                printStatSum("conflict impl count sum", sumConflictImplying);
#endif
                writeAsJson("approximateMemAllocated_megabytes", apprMemAllocated / 1.0e6);
                // TODO: re-add
                /*
                gpuRunner.printStats();
                reported.printStats();
                clauses.printStats();
                assigs.printStats();
                */
            }
        }
    } 
    nbprinted++;
}

void GpuMultiSolver::writeClausesInCnf() {
    SyncOut so;
    printf("c Writing clauses at %lf\n", realTimeSecSinceStart());
    gpuClauseSharer.writeClausesInCnf(stdout);
}

void GpuMultiSolver::configure() {
    for (int i = 1; i < helpedSolvers.size(); i++) {
        helpedSolvers[i]->randomizeFirstDescent = true;
        helpedSolvers[i]->adaptStrategies = (i % 2 == 0); // Just half of the cores are in adaptive mode
        helpedSolvers[i]->forceUnsatOnNewDescent = (i % 4 == 0); // Just half of adaptive cores have the unsat force
    }
}

long GpuMultiSolver::getStatSum(int stat) {
    long res = 0;
    std::lock_guard<std::mutex>lock(solversMutex);
    for(int i = 0; i < helpedSolvers.size(); i++) {
        if (helpedSolvers[i] != NULL) res += helpedSolvers[i]->stats[stat];
    }
    return res;
}


// for a cuda app, there's a crazy high amount of memory registered which
// isn't really used, so we don't count that
double GpuMultiSolver::actualCpuMemUsed() {
     return memUsed() - initMemUsed;
}

GpuMultiSolver::~GpuMultiSolver() {
    std::lock_guard<std::mutex>lock(solversMutex);
    for (int i = 0; i < helpedSolvers.size(); i++) {
        if (helpedSolvers[i] != NULL) {
            delete helpedSolvers[i];
        }
    }
}

}
