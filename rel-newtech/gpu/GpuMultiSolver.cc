/***************************************************************************************

MapleGpuShare, based on MapleLCMDistChronoBT-DL -- Copyright (c) 2020, Nicolas Prevot. Uses the GPU for clause sharing.

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
#include "../utils/System.h"
#include "../utils/JsonWriter.h"
#include "gpuShareLib/Utils.h"
#include "../core/Dimacs.h"
#include <thread> 
#include "Periodic.h"
#include "gpuShareLib/my_make_unique.h"
#include "../core/Finisher.h"
#include "../utils/Utils.h"
#include "../utils/JsonWriter.h"

namespace Minisat {

// There should only be one instance of this class.
// This class synchronizes the work of several SimpSolver
GpuMultiSolver::GpuMultiSolver(Finisher &_finisher, GpuShare::GpuClauseSharer &_gpuClauseSharer, 
        std::function<SimpSolver* (int threadId) > _solverFactory, int writeClausesPeriodSec,
        Verbosity _verb, double _maxMemory, int _gpuReduceDbPeriod, int _gpuReduceDbPeriodInc,
        JsonWriter *_writer, const GpuShare::Logger &_logger):
                gpuClauseSharer(_gpuClauseSharer),
                verb(_verb),
                gpuReduceDbPeriod(_gpuReduceDbPeriod),
                gpuReduceDbPeriodInc(_gpuReduceDbPeriodInc),
                maxMemory(_maxMemory),
                hasTriedToLowerCpuMemoryUsage(false),
                finisher(_finisher),
                solverFactory(_solverFactory),
                writer(_writer),
                logger(_logger) {
    periodicRunner = my_make_unique<PeriodicRunner>(realTimeSecSinceStart()); 
    periodicRunner->add(verb.writeStatsPeriodSec, std::function<void ()> ([&] () { 
        printStats();
    }));
    periodicRunner->add(writeClausesPeriodSec, std::function<void ()> ([&] () {
        writeClausesInCnf();
    }));
    
    solvers.growTo(1);
    solvers[0] = solverFactory(0);
}

void GpuMultiSolver::addClause_(vec<Lit>& lits) {
    assert(solvers.size() == 1);
    solvers[0]->addClause_(lits);
}

void GpuMultiSolver::addClause(const vec<Lit>& lits) {
    assert(solvers.size() == 1);
    solvers[0]->addClause(lits);
}

void launchSolver(std::mutex &mutex, SimpSolver*& solver, Finisher &finisher, lbool &result, const GpuShare::Logger &logger) {
    lbool status = solver->solve(false, true);
    // solvers which didn't find an answer are no longer useful, destroy them to free memory
    // But, if stopAllThreads is set, we probably still want to print stats for all solvers, so keep it
    if (status == l_Undef && !finisher.stopAllThreads) {
        SimpSolver *copy = solver;
        {
            std::lock_guard<std::mutex> lock(mutex);
            solver = NULL;
        }
        delete copy;
    }
    if (status != l_Undef) {
        assert(result == l_Undef || result == status);
        result = status;
    }
    SyncOut so;
    LOG(logger, 1, "c A thread is exiting");
}

lbool GpuMultiSolver::simplify() {
    int ret2 = solvers[0]->simplify();
    if(ret2) solvers[0]->eliminate(true);
    if (!solvers[0]->okay()) {
        LOG(logger, 1, "c Solved by unit propagation");
        return l_False;
    }
    return l_Undef;
}

lbool GpuMultiSolver::solve(int _cpuThreadCount) {
    cpuSolverCount = _cpuThreadCount;
    solvers.growTo(cpuSolverCount);
    finisher.stopAllThreadsAfterId = _cpuThreadCount;
    for (int i = 1; i < cpuSolverCount; i++) {
        solvers[i] = solverFactory(i);
        // We could be faster if we did this in multiple threads
        solvers[i]->copyClausesFrom(*solvers[0]);
        solvers[i]->use_simplification = false;
    }
    configure();

    LOG(logger, 1, "c |  all clones generated. Memory = " << memUsed() << "Mb");
    gpuClauseSharer.setCpuSolverCount(_cpuThreadCount);
    vec<std::thread> threads;
    long maxApprMemAllocated = -1;
    // Launching all solvers
    threads.growTo(solvers.size());
    lbool result = l_Undef;
    for(int i = 0; i < solvers.size(); i++) {
        threads[i] = std::thread(launchSolver, std::ref(solversMutex), std::ref(solvers[i]), std::ref(finisher), std::ref(result), std::ref(logger));
    }
    LOG(logger, 1, "All solvers launched");

    while (!finisher.stopAllThreads && finisher.stopAllThreadsAfterId > 0) {
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
            LOG(logger, 1, "All solvers launched");
            LOG(logger, 1, "c There is " << cpuMemUsed << " megabytes of memory used on cpu which is high, the limit is " << maxMemory <<", going to try reducing memory usage");
            // All the clauses on the GPU are also on the CPU, so limit the growth of their number
            gpuReduceDbPeriodInc = 0;
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
            for (int i = 0; i < solvers.size(); i++) {
                maxApprMemAllocated += solvers[i]->getApproximateMemoryAllocated();
            }
            LOG(logger, 1, "c There is " << cpuMemUsed << " megabytes of memory used on the cpu when the limit is " << maxMemory << ". Going to kill threads to reduce memory usage");
            LOG(logger, 1, "c The approximate memory allocated is " << maxApprMemAllocated  / 1.0e6);
            finisher.stopAllThreadsAfterId = std::max(solvers.size() - 1, 1);
        }
        if (maxApprMemAllocated > 0)
        for (int i = 0; i < finisher.stopAllThreadsAfterId; i++) {
            long apprMemAllocated = 0;
            for (int i = 0; i < finisher.stopAllThreadsAfterId; i++) {
                std::lock_guard<std::mutex> lock(solversMutex);
                if (solvers[i] != NULL) {
                    apprMemAllocated += solvers[i]->getApproximateMemoryAllocated();
                    if (apprMemAllocated > maxApprMemAllocated && i > 0) {
                        finisher.stopAllThreadsAfterId = i;
                    }
                }
            }
        }
    }
    if (verb.global > 0) {
        LOG(logger, 1, "c printing final stats");
        printStats();
    }

    for(int i = 0; i < threads.size(); i++) { // Wait for all threads to finish
        threads[i].join();
    }
    return result;
}

vec<lbool>& GpuMultiSolver::getModel() {
    int threadId = finisher.oneThreadIdWhoFoundAnAnswer;
    vec<lbool> &mod = solvers[threadId]->model;
    // The information about eliminated clauses is only in solvers[0]
    if (threadId != 0) solvers[0]->extendModel(mod);
    return mod;
}

void GpuMultiSolver::printStats() {
    if (writer == NULL) return;
    JsonWriter &wr(*writer);
    SyncOut so;
    static int nbprinted = 1;

    size_t freeGpuMem;
    size_t totalGpuMem;
    long apprMemAllocated = 0;
    gpuClauseSharer.getGpuMemInfo(freeGpuMem, totalGpuMem);
    {
        JObj jobj(wr);
        wr.writeJsonString("type", "periodicStats");
        wr.write("cpuTime", cpuTimeSec());
        wr.write("realTime", realTimeSecSinceStart());
        {
            wr.writeJsonField("solverStats");
            {
                JArr jarr(wr);
                std::lock_guard<std::mutex> lock(solversMutex);
                for (int i = 0; i < solvers.size(); i++) {
                    if (solvers[i] != NULL) {
                        solvers[i]->printStats(wr, false);
                        apprMemAllocated += solvers[i]->getApproximateMemoryAllocated();
                    }
                }
            }
            wr.writeJsonField("globalStats");
            {
                JObj jo(wr);
                wr.write("cpuMemUsed_megabytes", actualCpuMemUsed());
                wr.write("gpuMemUsed_megabytes", (float) (totalGpuMem - freeGpuMem) / 1.0e6);
                wr.write("approximateMemAllocated_megabytes", apprMemAllocated / 1.0e6);
                for (int i = 0; i < gpuClauseSharer.getGlobalStatCount(); i++) {
                    GpuShare::GlobalStats gs = static_cast<GpuShare::GlobalStats>(i);
                    wr.write(gpuClauseSharer.getGlobalStatName(gs), gpuClauseSharer.getGlobalStat(gs));
                }
            }
        }
    }
    nbprinted++;
}

void GpuMultiSolver::writeClausesInCnf() {
    SyncOut so;
    printf("c Writing clauses at %lf", realTimeSecSinceStart());
    gpuClauseSharer.writeClausesInCnf(stdout);
}

void GpuMultiSolver::configure() {
    for (int i = 1; i < solvers.size(); i++) {
        // TODO: change it here
        /*
        solvers[i]->randomizeFirstDescent = true;
        solvers[i]->adaptStrategies = (i % 2 == 0); // Just half of the cores are in adaptive mode
        solvers[i]->forceUnsatOnNewDescent = (i % 4 == 0); // Just half of adaptive cores have the unsat force
        */
    }
}


// for a cuda app, there's a crazy high amount of memory registered which
// isn't really used, so we don't count that
double GpuMultiSolver::actualCpuMemUsed() {
     return memUsed() - initMemUsed;
}

GpuMultiSolver::~GpuMultiSolver() {
    std::lock_guard<std::mutex>lock(solversMutex);
    for (int i = 0; i < solvers.size(); i++) {
        if (solvers[i] != NULL) {
            delete solvers[i];
        }
    }
}

}
