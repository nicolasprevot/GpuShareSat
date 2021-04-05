#include "Helper.cuh"
#include "GpuClauseSharerImpl.cuh"
#include "GpuClauseSharer.h"
#include "GpuUtils.cuh"
#include "Clauses.cuh"
#include "Reported.cuh"
#include "GpuRunner.cuh"
#include "Assigs.cuh"
#include <thread>

namespace GpuShare {

extern size_t maxPageLockedMem;

GpuClauseSharer* makeGpuClauseSharerPtr(GpuClauseSharerOptions opts, std::function<void (const std::string &str)> logFunc) {
    return new GpuClauseSharerImpl(opts, logFunc);
}

void writeMessageAndThrow(const char *message) {
    fprintf(stderr, "%s", message);
    THROW();
}

GpuClauseSharerImpl::GpuClauseSharerImpl(GpuClauseSharerOptions _opts, std::function<void (const std::string &str)> logFunc): logger {_opts.verbosity, logFunc} {
    // It can be necessary for debugging if we print a lot
    // 100 Megs
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1048576 * 100);

// assumes the enum values start at 0
#define X(v) globalStatNames.push_back(#v);
#include "GlobalStats.h"
#undef X

// assumes the enum values start at 0
#define X(v) oneSolverStatNames.push_back(#v);
#include "OneSolverStats.h"
#undef X
    globalStats.resize(globalStatNames.size());

    opts = _opts;
    if (opts.minGpuLatencyMicros < 0) opts.minGpuLatencyMicros = 50;
    if (opts.clauseActivityDecay < 0) opts.clauseActivityDecay = 0.99999;
    if (opts.maxPageLockedMemory < 0) {
        size_t freeGpuMem, totalGpuMem;
        getGpuMemInfo(freeGpuMem, totalGpuMem);
        // This is a bit wrong because page locked memory is host memory. However, there is no simple API which gives us the physical
        // memory on the host while there is one for the device.
        opts.maxPageLockedMemory = totalGpuMem / 3;
    }
    if (opts.gpuBlockCountGuideline < 0) {
        cudaDeviceProp props;
        exitIfError(cudaGetDeviceProperties(&props, 0), POSITION);
        opts.gpuBlockCountGuideline = props.multiProcessorCount * 2;

        LOG(logger, 1, "c Setting block count guideline to " << opts.gpuBlockCountGuideline << " (twice the number of multiprocessors)\n");
    }
    if (opts.gpuThreadsPerBlockGuideline < 0) opts.gpuThreadsPerBlockGuideline = 512;

    if (opts.clauseActivityDecay >= 1) writeMessageAndThrow("Clause activity decay must be strictly smaller than 1");

    if (opts.initReportCountPerCategory < 0) opts.initReportCountPerCategory = 10;

    if (opts.initReportCountPerCategory == 0) writeMessageAndThrow("initReportCountPerCategory must not be 0");
    if (opts.gpuThreadsPerBlockGuideline == 0) writeMessageAndThrow("gpuThreadsPerBlockGuideline must not be 0");
    if (opts.gpuBlockCountGuideline == 0) writeMessageAndThrow("gpuBlockCountGuideline must not be 0");

    varCount = 0;
    GpuDims gpuDims {opts.gpuBlockCountGuideline, opts.gpuThreadsPerBlockGuideline};

    maxPageLockedMem = opts.maxPageLockedMemory;
    assigs = my_make_unique<HostAssigs>(gpuDims, logger);  
    clauses = my_make_unique<HostClauses>(gpuDims, opts.clauseActivityDecay, true, globalStats, logger);
    reported = my_make_unique<Reported>(*clauses, oneSolverStats);
    gpuRunner = my_make_unique<GpuRunner>(*clauses, *assigs, *reported, gpuDims, opts.quickProf, opts.initReportCountPerCategory, sp.get(), globalStats, logger);

}

void GpuClauseSharerImpl::setVarCount(int newVarCount) {
    varCount = newVarCount;
    assigs->setVarCount(newVarCount, sp.get());
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
    for (int i = c; i < solverCount; i++) oneSolverStats[i].resize(oneSolverStatNames.size(), 0);
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

long GpuClauseSharerImpl::addClause(int solverId, int *lits, int count) {
    GpuClauseId clId = clauses->addClause(MinHArr<Lit>(count, (Lit*) lits), count);
    if (solverId != -1) reported->clauseWasAdded(solverId, clId);
    return clId;
}

bool GpuClauseSharerImpl::trySetSolverValues(int solverId, int *lits, int count) {
    OneSolverAssigs& sAssigs = assigs->getAssigs(solverId);
    MinHArr<Lit> litsToSet(count, (Lit*) lits);
    bool success = false;
    sAssigs.enterLock();
    if (sAssigs.isAssignmentAvailableLocked()) {
        unsetPendingLocked(solverId);
        for (int i = 0; i < litsToSet.size(); i++) {
            sAssigs.setVarLocked(var(litsToSet[i]), sign(litsToSet[i]) ? gl_False : gl_True);
        }
        oneSolverStats[solverId][varUpdatesSentToGpu] += litsToSet.size();
        success = true;
    } else {
        oneSolverStats[solverId][failuresToFindAssig]++;
    }
    sAssigs.exitLock();
    return success;
}

void GpuClauseSharerImpl::unsetPendingLocked(int solverId) {
    std::vector<Lit> &unset = toUnset[solverId];
    OneSolverAssigs& sAssigs = assigs->getAssigs(solverId);
    for (int i = 0; i < unset.size(); i++) {
        sAssigs.setVarLocked(var(unset[i]), gl_Undef);
    }
    oneSolverStats[solverId][varUpdatesSentToGpu] += unset.size();
    unset.clear();
}

void GpuClauseSharerImpl::unsetSolverValues(int solverId, int *lits, int count) {
    OneSolverAssigs& sAssigs = assigs->getAssigs(solverId);
    MinHArr<Lit> litsToUnset(count, (Lit*) lits);
    sAssigs.enterLock();
    if (sAssigs.isAssignmentAvailableLocked()) {
        unsetPendingLocked(solverId);
        for (int i = 0; i < litsToUnset.size(); i++) {
            sAssigs.setVarLocked(var(litsToUnset[i]), gl_Undef);
        }
        oneSolverStats[solverId][varUpdatesSentToGpu] += litsToUnset.size();
    } else {
        std::vector<Lit> &unset = toUnset[solverId];
        int start = unset.size();
        unset.resize(start + litsToUnset.size());
        memcpy(&unset[start], &litsToUnset[0], sizeof(Lit) * litsToUnset.size());
    }
    sAssigs.exitLock();
}

long GpuClauseSharerImpl::trySendAssignment(int solverId) {
    OneSolverAssigs& sAssigs = assigs->getAssigs(solverId);
    long result = -1;
    sAssigs.enterLock();
    if (sAssigs.isAssignmentAvailableLocked()) {
        result = sAssigs.assignmentDoneLocked();
        oneSolverStats[solverId][assigsSentToGpu]++;
        reported->assigWasSent(solverId, result);
    } else {
        oneSolverStats[solverId][failuresToFindAssig]++;
    }
    sAssigs.exitLock();
    return result;
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
    if (stat == clauseTestsOnAssigs) return gpuRunner->getClauseTestsOnAssigs();
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

void GpuClauseSharerImpl::getCurrentAssignment(int solverId, uint8_t *assig) { 
    assigs->getAssigs(solverId).getCurrentAssignment(assig);
    for (int i = 0; i < toUnset[solverId].size(); i++) {
        Var v = var(toUnset[solverId][i]);
        assig[v] = toInt(gl_Undef);
    }
}

long GpuClauseSharerImpl::getLastAssigAllReported(int cpuSolverId) {
    return reported->getLastAssigAllReported(cpuSolverId);
}

GpuClauseSharerImpl::~GpuClauseSharerImpl() {
}

}
