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
#include "GpuMultiSolver.cuh"
#include "GpuRunner.cuh"
#include "GpuHelpedSolver.cuh"
#include "Reported.cuh"
#include "Assigs.cuh"
#include "Clauses.cuh"
#include "utils/System.h"
#include "utils/Utils.h"
#include "satUtils/Dimacs.h"
#include <pthread.h>
#include "utils/Periodic.h"
#include "my_make_unique.h"

namespace Glucose {

// There should only be one instance of this class.
// This class synchronizes the work of several GpuHelpedSolver
GpuMultiSolver::GpuMultiSolver(GpuRunner &_gpuRunner, Reported &_reported, Finisher &_finisher, HostAssigs &_assigs, HostClauses &_clauses,
        std::function<std::unique_ptr<GpuHelpedSolver> (int threadId, OneSolverAssigs&) > _solverFactory, int varCount, int writeClausesPeriodSec,
        double _initMemUsed, double _maxMemory):
                gpuRunner(_gpuRunner),
                reported(_reported),
                finisher(_finisher),
                assigs(_assigs),
                clauses(_clauses),
                solverFactory(_solverFactory),
                verb(0, 0, 0),
                initMemUsed(_initMemUsed),
                maxMemory(_maxMemory),
                hasTriedToLowerCpuMemoryUsage(false) {
    periodicRunner = my_make_unique<PeriodicRunner>(realTimeSecSinceStart()); 
    periodicRunner->add(5, std::function<void ()> ([&] () { 
        printStats();
    }));
    periodicRunner->add(writeClausesPeriodSec, std::function<void ()> ([&] () {
        writeClauses();
    }));
    
    
    float memBefore = memUsed();
    helpedSolvers.growTo(1);
    helpedSolvers[0] = solverFactory(0, assigs.getAssigs(0));
    for (int v = 0; v < varCount; v++) {
        helpedSolvers[0]->newVar();
    }
    memUsedCreateOneSolver =  memUsed() - memBefore;
}

void GpuMultiSolver::addClause_(vec<Lit>& lits) {
    assert(helpedSolvers.size() == 1);
    helpedSolvers[0]->addClause_(lits);
}

void* launchSolver(void *arg) {
    GpuHelpedSolver *s = (GpuHelpedSolver *) arg;
    s->solve();

    pthread_exit(NULL);
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
    for (int i = 1; i < cpuSolverCount; i++) {
        helpedSolvers[i] = my_make_unique<GpuHelpedSolver>(*helpedSolvers[0], i, assigs.getAssigs(i));
    }
    configure();

    printf("c |  all clones generated. Memory = %6.2fMb.                                                             |\n", memUsed());
    printf("c ========================================================================================================|\n");

    reported.setSolverCount(_cpuThreadCount);
    pthread_attr_t thAttr;
    pthread_attr_init(&thAttr);
    pthread_attr_setdetachstate(&thAttr, PTHREAD_CREATE_JOINABLE);
    vec<pthread_t*> threads;
    // Launching all solvers
    for(int i = 0; i < helpedSolvers.size(); i++) {
        pthread_t *pt = (pthread_t *) malloc(sizeof(pthread_t));
        threads.push(pt);
        pthread_create(pt, &thAttr, &launchSolver, (void *) helpedSolvers[i].get());
    }
    printf("All solvers launched\n");

    while (true) {
        if (finisher.hasCanceledOrFinished()) {
            break;
        }
        periodicRunner->maybeRun(realTimeSecSinceStart());
        gpuRunner.execute();
        double cpuMemUsed = actualCpuMemUsed();
        if (!hasTriedToLowerCpuMemoryUsage && cpuMemUsed > maxMemory) {
            // We're not very strict about memory usage on the cpu. Reason
            // is that if we use more than physical memory, it will swap
            // It's very different for gpu memory usage where there's no swap
            // so it crashes if we go over the limit
            printf("There is %lf megabytes of memory used on cpu which is higher than the limit of %lf, going to try reducing memory usage\n", cpuMemUsed, maxMemory);
            clauses.tryReduceCpuMemoryUsage();
            for (int i = 0; i < helpedSolvers.size(); i++) {
                helpedSolvers[i]->tryReduceCpuMemoryUsage();
            }
            hasTriedToLowerCpuMemoryUsage = true;
        }
    }
    printf("c printing final stats\n");
    printStats();

    for(int i = 0; i < threads.size(); i++) { // Wait for all threads to finish
        pthread_join(*threads[i], NULL);
    }
    if (finisher.isCanceled()) { return l_Undef; }
    int whoHasFinished = finisher.getOneWhoHasFinished();
    return helpedSolvers[whoHasFinished]->getStatus();
}

vec<lbool>& GpuMultiSolver::getModel() {
    return helpedSolvers[finisher.getOneWhoHasFinished()]->model;
}

void GpuMultiSolver::printStatSum(const char* name, int stat) {
    printf("c %s: %ld\n", name, getStatSum(stat));
}

void GpuMultiSolver::printStats() {
    SyncOut so;
    static int nbprinted = 1;

    size_t freeGpuMem;
    size_t totalGpuMem;
    exitIfError(cudaMemGetInfo(&freeGpuMem, &totalGpuMem), POSITION);

    {
        JStats jstats;
        writeJsonString("type", "periodicStats");
        writeAsJson("cpuTime", cpuTimeSec());
        writeAsJson("realTime", realTimeSecSinceStart());
        {
            writeJsonField("solverStats");
            {
                JArr jarr;
                for (int i = 0; i < helpedSolvers.size(); i++) {
                    helpedSolvers[i]->printStats();
                }
            }
            writeJsonField("globalStats");
            {
                JObj jo;
                writeAsJson("cpuMemUsed_megabytes", actualCpuMemUsed());
                writeAsJson("gpuMemUsed_megabytes", (float) (totalGpuMem - freeGpuMem) / 1e6);
#ifdef KEEP_IMPL_COUNT
                printStatSum("conflict impl count sum", sumConflictImplying);
#endif
                gpuRunner.printStats();
                reported.printStats();
                clauses.printStats();
                assigs.printStats();
            }
        }
    } 
    nbprinted++;
}

void GpuMultiSolver::writeClauses() {
    SyncOut so;
    printf("c Writing clauses at %lf\n", realTimeSecSinceStart());
    printf("p cnf %d %d\n", helpedSolvers[0]->nVars(), clauses.getClauseCount());
    vec<Lit> lits;
    int gpuAssigId;
    for (int clSize = 1; clSize <= MAX_CL_SIZE; clSize++) {
        int count = clauses.getClauseCount(clSize);
        for (int clIdInSize = 0; clIdInSize < count; clIdInSize++) {
            GpuCref gpuCref {clSize, clIdInSize};
            clauses.getClause(lits, gpuAssigId, gpuCref);
            writeClause(lits);
        }
    }
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
    for(int i = 0; i < helpedSolvers.size(); i++) {
        res += helpedSolvers[i] -> stats[stat];
    }
    return res;
}


// for a cuda app, there's a crazy high amount of memory registered which
// isn't really used, so we don't count that
double GpuMultiSolver::actualCpuMemUsed() {
     return memUsed() - initMemUsed;
}

}
