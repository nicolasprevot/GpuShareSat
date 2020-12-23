#include "Helper.cuh"
#include "GpuClauseSharerImpl.cuh"
#include "GpuClauseSharer.h"
#include "GpuUtils.cuh"
#include "Clauses.cuh"
#include "Reported.cuh"
#include "GpuRunner.cuh"
#include "Assigs.cuh"
#include "utils/System.h"
#include <thread>

namespace Glucose {
GpuClauseSharerImpl::GpuClauseSharerImpl(GpuClauseSharerOptions _opts, /* TODO: we should be able to increase it */ int _varCount) {
// assumes the enum values start at 0
#define X(v) globalStatNames.push(#v);
#include "GlobalStats.h"
#undef X

// assumes the enum values start at 0
#define X(v) oneSolverStatNames.push(#v);
#include "OneSolverStats.h"
#undef X
    globalStats.resize(globalStatNames.size());

    opts = _opts;
    varCount = _varCount;
    GpuDims gpuDims {0, 0};
    if (opts.gpuBlockCountGuideline > 0) {
        gpuDims.blockCount = opts.gpuBlockCountGuideline;
    } else {
        cudaDeviceProp props;
        exitIfError(cudaGetDeviceProperties(&props, 0), POSITION);
        gpuDims.blockCount = props.multiProcessorCount * 2;
        if (opts.verbosity > 0) printf("c Setting block count guideline to %d (twice the number of multiprocessors)\n", gpuDims.blockCount);
    }
    gpuDims.threadsPerBlock = opts.gpuThreadsPerBlockGuideline;
    assigs = my_make_unique<HostAssigs>(varCount, gpuDims);  
    clauses = my_make_unique<HostClauses>(gpuDims, opts.clauseActivityDecay, true);
    reported = my_make_unique<Reported>(*clauses, oneSolverStats);
    gpuRunner = my_make_unique<GpuRunner>(*clauses, *assigs, *reported, gpuDims, opts.quickProf, _opts.initReportCountPerCategory, sp.get());

}

void GpuClauseSharerImpl::gpuRun() {
    long timeMicrosBegining = realTimeMicros();
    gpuRunner->wholeRun(true);
    long timePassedMicros = realTimeMicros() - timeMicrosBegining;
    if (timePassedMicros < opts.minGpuLatencyMicros) {
        // reason: at the beginning, there aren't many clauses
        // we'd just loop burning cpu and copying clauses. So make sure that the loop takes at least
        // a certain amount of time
        std::this_thread::sleep_for(std::chrono::microseconds(opts.minGpuLatencyMicros - timePassedMicros));
    }
}

void GpuClauseSharerImpl::setCpuSolverCount(int solverCount) {
    assigs->growSolverAssigs(solverCount);
    reported->setSolverCount(solverCount);
    toUnset.resize(solverCount);
    int c = oneSolverStats.size();
    oneSolverStats.resize(solverCount);
    for (int i = c; i < solverCount; i++) oneSolverStats[i].growToInit(oneSolverStatNames.size(), 0);
}

void GpuClauseSharerImpl::reduceDb() {
    gpuRunner->wholeRun(false);
    // we can't reduce db if there are runs in flight since what they return would not point to
    // the right clause any more
    clauses->reduceDb(sp.get());
}

bool GpuClauseSharerImpl::hasRunOutOfGpuMemoryOnce() {
    return gpuRunner->getHasRunOutOfGpuMemoryOnce();
}

long GpuClauseSharerImpl::getAddedClauseCount() {
    return clauses->getAddedClauseCount();
}

long GpuClauseSharerImpl::getAddedClauseCountAtLastReduceDb() {
    return clauses->getAddedClauseCountAtLastReduceDb();
}

long GpuClauseSharerImpl::addClause(int *lits, int count) {
    return clauses->addClause(MinHArr<Lit>(count, (Lit*) lits), count);
}

bool GpuClauseSharerImpl::trySetSolverValues(int solverId, int *lits, int count) {
    OneSolverAssigs& sAssigs = assigs->getAssigs(solverId);
    MinHArr<Lit> litsToSet(count, (Lit*) lits);
    bool success = false;
    sAssigs.enterLock();
    if (sAssigs.isAssignmentAvailableLocked()) {
        unsetPending(solverId);
        for (int i = 0; i < litsToSet.size(); i++) {
            sAssigs.setVarLocked(var(litsToSet[i]), sign(litsToSet[i]) ? l_False : l_True);
        }
        success = true;
    }
    sAssigs.exitLock();
    return success;
}

void GpuClauseSharerImpl::unsetPending(int solverId) {
    vec<Lit> &unset = toUnset[solverId];
    OneSolverAssigs& sAssigs = assigs->getAssigs(solverId);
    for (int i = 0; i < unset.size(); i++) {
        sAssigs.setVarLocked(var(unset[i]), l_Undef);
    }
    unset.clear();
}

void GpuClauseSharerImpl::unsetSolverValues(int solverId, int *lits, int count) {
    OneSolverAssigs& sAssigs = assigs->getAssigs(solverId);
    MinHArr<Lit> litsToUnset(count, (Lit*) lits);
    sAssigs.enterLock();
    if (sAssigs.isAssignmentAvailableLocked()) {
        unsetPending(solverId);
        for (int i = 0; i < litsToUnset.size(); i++) {
            sAssigs.setVarLocked(var(litsToUnset[i]), l_Undef);
        }
    } else {
        vec<Lit> &unset = toUnset[solverId];
        unset.resize(litsToUnset.size());
        memcpy(&unset[0], &litsToUnset[0], sizeof(Lit) * litsToUnset.size());
    }
    sAssigs.exitLock();
}

bool GpuClauseSharerImpl::trySendAssignment(int solverId) {
    OneSolverAssigs& sAssigs = assigs->getAssigs(solverId);
    bool success = false;
    sAssigs.enterLock();
    if (sAssigs.isAssignmentAvailableLocked()) {
        sAssigs.assignmentDoneLocked();
        success = true;
    }
    sAssigs.exitLock();
    return success;
}

bool GpuClauseSharerImpl::popReportedClause(int solverId, int* &lits, int &count, long &gpuClauseId) {
    MinHArr<Lit> litsArr;
    GpuClauseId clId;
    if (reported->popReportedClause(solverId, litsArr, clId)) {
        lits = (int*) litsArr.getPtr();
        count = litsArr.size();
        gpuClauseId = clId;
        return true;
    }
    return false;
}

void GpuClauseSharerImpl::getGpuMemInfo(size_t &free, size_t &total) {
    exitIfError(cudaMemGetInfo(&free, &total), POSITION);
}

void GpuClauseSharerImpl::writeClausesInCnf(FILE *file) {
    clauses->writeClausesInCnf(file, varCount);
}

int GpuClauseSharerImpl::getGlobalStatCount() {
    return globalStatNames.size();
}

long GpuClauseSharerImpl::getGlobalStat(GlobalStats stat) {
    return globalStats[stat];
}

const char* GpuClauseSharerImpl::getGlobalStatName(GlobalStats stat) {
    return globalStatNames[stat];
}

int GpuClauseSharerImpl::getOneSolverStatCount() {
    return oneSolverStatNames.size();
}

long GpuClauseSharerImpl::getOneSolverStat(int solverId, OneSolverStats stat) {
    return oneSolverStats[solverId][stat];
}

const char* GpuClauseSharerImpl::getOneSolverStatName(OneSolverStats stat) {
    return oneSolverStatNames[stat];
}

}
