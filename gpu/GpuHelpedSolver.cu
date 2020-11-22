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
#include "GpuHelpedSolver.cuh"
#include "Reported.cuh"
#include "Assigs.cuh"
#include "utils/Utils.h"
#include "Helper.cuh"
#include "Helper.cuh"
#include "Clauses.cuh"

#include <chrono>
#include <thread>
#include <sstream>
#include <mutex>

#define PRINT_ALOT 0

namespace Glucose {

GpuHelpedSolver::GpuHelpedSolver(const GpuHelpedSolver &other, int _cpuThreadId, OneSolverAssigs &_assigsForGpu) :
        SimpSolver(other, _cpuThreadId), reported(other.reported),
        trailCopiedUntil(other.trailCopiedUntil), changedCount(other.changedCount), status(other.status),
        hClauses(other.hClauses), nextAssigPos(other.nextAssigPos),
        lastReduceDbCount(other.lastReduceDbCount), conflictsLastReduceDb(other.conflictsLastReduceDb),
        seenAllReportsUntil(other.seenAllReportsUntil),
        params(other.params), assigsForGpu(_assigsForGpu),
        needToReduceCpuMemoryUsage(other.needToReduceCpuMemoryUsage) {

}

GpuHelpedSolver::GpuHelpedSolver(Reported &_reported, Finisher &_finisher,
        HostClauses &clauses, int _cpuThreadId,
        GpuHelpedSolverParams _params, OneSolverAssigs &_assigsForGpu) :
        reported(_reported), status(
        l_Undef), hClauses(clauses),
        trailCopiedUntil(0), changedCount(0), SimpSolver(_cpuThreadId, _finisher),
        seenAllReportsUntil(0),
        conflictsLastReduceDb(0),
        nextAssigPos(0),
        params(_params),
        assigsForGpu(_assigsForGpu),
        needToReduceCpuMemoryUsage(false) {
        // for the first gpu stat
        stats.push(0);
#define X(v) stats.push(0);
#include "GpuSolverStats.h"
#undef X
	insertStatNames();
}

void GpuHelpedSolver::insertStatNames() {
    statNames[nbexportedgpu] = std::string("nbexportedgpu");
#define X(v) statNames[v] = std::string(#v);
#include "GpuSolverStats.h"
#undef X
}

// The reason for having foundEmptyClause (and not just having a thread finish once it finds the empty clause) is that:
// the caller needs to return a status. (true, false, undef). If we find an empty clause, we have no
// other way of returning the status false (except for setting the status ourselves, maybe, but since
// this code doesn't know about status, status is set at a higher level, that's not great)
// Even if the gpu doesn't have the empty clause: it may have the clause ~a when we know a to be true
// In this case, we can't return a cref conflict for ~a because a clause with a cref can't have a size of 1
CRef GpuHelpedSolver::gpuImportClauses(bool& foundEmptyClause) {
    foundEmptyClause = false;

    ClauseBatch* clBatch;
    CRef confl = CRef_Undef;
    int decisionLevelAtConflict = -1;
    while (reported.getIncrReportedClauses(cpuThreadId, clBatch)) {
        if (params.import) {
            handleClBatch(*clBatch, confl, decisionLevelAtConflict, foundEmptyClause);
        }
        while (reported.getOldestClauses(cpuThreadId, clBatch)
                && clBatch->assigWhichKnowsAboutThese <= seenAllReportsUntil) {
            auto &clDatas = clBatch->getClauseDatas();
            for (int i = 0; i < clDatas.size(); i++) {
                clausesToNotImportAgain.erase(clDatas[i].gpuClauseId);
            }
            reported.removeOldestClauses(cpuThreadId);
        }
    }

    if (decisionLevel() == decisionLevelAtConflict) {
        return confl;
    }
    return CRef_Undef;
}

void GpuHelpedSolver::handleClBatch(ClauseBatch &clBatch, CRef &conflict, int &decisionLevelAtConflict, bool &foundEmptyClause) {
    clBatch.assigWhichKnowsAboutThese = nextAssigPos;
    MinHArr<Lit> lits;
    GpuClauseId gpuClauseId;
    while (clBatch.popClause(lits, gpuClauseId)) {
        handleReportedClause(lits, gpuClauseId, conflict, decisionLevelAtConflict, foundEmptyClause);
    }
    seenAllReportsUntil = clBatch.assigIds.startAssigId + clBatch.assigIds.assigCount;
    stats[nbAssigNoReport] += clBatch.assigIds.assigCount - countBitsSet(clBatch.hadSomeReported);
}

void GpuHelpedSolver::sendClauseToGpu(vec<Lit> &lits, int lbd) {
    hClauses.addClause(lits, lbd);
    if (lits.size() == 1) {
        stats[nbExportedUnit]++;
    } else {
        stats[nbexportedgpu]++;
    }
}

void GpuHelpedSolver::handleReportedClause(MinHArr<Lit> &lits, GpuClauseId &gpuClauseId,
        CRef &conflict, int &decisionLevelAtConflict, bool &foundEmptyClause) {
    stats[nbReported] ++;

    // Check if we can import on the cpu the reported clause
    // Issue is: there's a possibility that the exact same clause has been reported several times
    // for different assignments, we don't want to import it twice in this case
    // Note: for one gpu run, a clause will be reported at most once for a solver,
    // but it may have been already reported in the previous gpu run
    bool canAdd = true;
    if (clausesToNotImportAgain.find(gpuClauseId) != clausesToNotImportAgain.end()) {
        canAdd = false;
#ifdef PRINT_DETAILS_CLAUSES
        SyncOut so;
        std::cout << "not_learn: thread " << cpuThreadId << " cl size " << lits.size();
#ifdef PRINT_DETAILS_LITS
        std::cout << " " << lits;
#endif
        std::cout << std::endl;
#endif
    } else {
        clausesToNotImportAgain.insert(gpuClauseId);
    }
    if (canAdd) {
        CRef cr = insertAndLearnClause(lits, foundEmptyClause);
        if (cr != CRef_Undef) {
            decisionLevelAtConflict = decisionLevel();
            conflict = cr;
        }
    }
}

void GpuHelpedSolver::unsetFromTrailForGpuLocked(int level) {
    // if level is decisionLevel, there's no trail_lim for it
    if (level < decisionLevel()) {
        for (int i = trail_lim[level]; i < trailCopiedUntil; i++) {
            assigsForGpu.setVarLocked(var(trail[i]), l_Undef);
        }
        // doing this because trail may actually be copied until less than that
        trailCopiedUntil = min(trailCopiedUntil, trail_lim[level]);
    }

}

void GpuHelpedSolver::copyTrailForGpuLocked(int level) {
    int max;
    // Note that we may unset variables from two sources:
    // - Those from toUnsetFromAssigsForGpu (if cancelUntil happened before)
    // - Some more to get to level, which may not be the current decision level
    for (int i = 0; i < toUnsetFromAssigsForGpu.size(); i++) {
        assigsForGpu.setVarLocked(toUnsetFromAssigsForGpu[i], l_Undef);
    }
    toUnsetFromAssigsForGpu.clear();
    if (level < decisionLevel()) {
        unsetFromTrailForGpuLocked(level);
        max = trail_lim[level];
    } else {
        max = trail.size();
    }
    while (trailCopiedUntil < max) {
        Lit p = trail[trailCopiedUntil];
        Var v = var(p);
        lbool val = value(v);
        assert(val != l_Undef);
        assigsForGpu.setVarLocked(v, val);
        trailCopiedUntil++;
        changedCount++;
    }
    assigsForGpu.assignmentDoneLocked();
}

void GpuHelpedSolver::cancelUntil(int level) {
    if (level < decisionLevel()) {
        for (int i = trail_lim[level]; i < trailCopiedUntil; i++) {
            toUnsetFromAssigsForGpu.push(var(trail[i]));
        }
        // doing this because trail may actually be copied until less than that
        trailCopiedUntil = min(trailCopiedUntil, trail_lim[level]);
    }
    Solver::cancelUntil(level);
}

void GpuHelpedSolver::foundConflict(vec<Lit> &learned, int lbd) {
    foundConflictInner(decisionLevel() - 1, learned, lbd);
}

void GpuHelpedSolver::foundConflictInner(int level, vec<Lit> &learned, int lbd) {
    int assigId = copyAssigsForGpuInner(level);
    if (lbd >= 0) {
        sendClauseToGpu(learned, lbd);
    }
}

// here only for testing
void GpuHelpedSolver::copyAssigsForGpu(int level) {
    vec<Lit> lits;
    // -1 indicates that we don't care about the clause
    foundConflictInner(level, lits, -1);
}

int GpuHelpedSolver::copyAssigsForGpuInner(int level) {
    assigsForGpu.enterLock();
    int res;
    if (assigsForGpu.isAssignmentAvailableLocked()) {
        copyTrailForGpuLocked(level);
        res = nextAssigPos;
        nextAssigPos++;
    }
    else {
        stats[nbFailureFindAssignment]++;
        res = -1;
    }
    assigsForGpu.exitLock();
    return res;
}

// finds the lit with the largest level among lits from start to the end and put it in start
void GpuHelpedSolver::findLargestLevel(MinHArr<Lit>& lits, int start) {
    int size = lits.size();
    for (int i = start + 1; i < size; i++) {
        if (litLevel(lits[i]) > litLevel(lits[start])) {
            std::swap(lits[i], lits[start]);
        }
    }
}

// Learns the clause, and attach it
// May lead to canceling to backtracking if this clause leads to an implication at a level strictly lower
// than the current level
// returns a non-undef cref if this clause is in conflict
CRef GpuHelpedSolver::insertAndLearnClause(MinHArr<Lit> &lits, bool &foundEmptyClause) {

    // if only one literal isn't false, it will be in 0 of lits
    findLargestLevel(lits, 0);
    // if only two literals aren't set, they will be in 0 and 1 of lits
    findLargestLevel(lits, 1);
    int size = lits.size();
    if (size == 0) {
        stats[nbImportedValid]++;
        foundEmptyClause = true;
        return CRef_Undef;
    }
    if (size == 1) {
        lbool val = value(lits[0]);
        if (val == l_False && level(var(lits[0])) == 0) {
            foundEmptyClause = true;
            stats[nbImportedValid]++;
            return CRef_Undef;
        }
        // we've learned a unary clause we already knew
        if (val == l_True && level(var(lits[0])) == 0) {
            return CRef_Undef;
        }
        stats[nbImportedValid]++;
        stats[nbImportedUnit]++;
        cancelUntil(0);
        uncheckedEnqueue(lits[0]);
        return CRef_Undef;
    }
    if (value(lits[1]) != l_False) {
        // two literals not set: we can just learn the clause
        assert(value(lits[0]) != l_False);
        addLearnedClause(lits);
        return CRef_Undef;
    }
    if (value(lits[0]) == l_True
            && level(var(lits[0])) <= level(var(lits[1]))) {
        // We're implying a literal at a level <= to what it is already. Just learn the clause, it's not doing anything now, though
        addLearnedClause(lits);
        return CRef_Undef;
    }
    if ((value(lits[0]) == l_False)
            && (level(var(lits[1])) == level(var(lits[0])))) {
        // conflict
#ifdef DEBUG
        for (int i = 0; i < lits.size(); i++) {
            assert(value(lits[i]) == l_False);
        }
#endif
        assert(value(lits[1]) == l_False);
        stats[nbImportedValid]++;
        cancelUntil(level(var(lits[1])));
        qhead = trail.size();
        return addLearnedClause(lits);
    }
    // lit 0 is implied by the rest, may currently be undef or false
#ifdef DEBUG
    for (int i = 1; i < lits.size(); i++) {
        assert(value(lits[i]) == l_False);
    }
#endif
    CRef cr = addLearnedClause(lits);
    cancelUntil(level(var(lits[1])));
    stats[nbImportedValid]++;
    uncheckedEnqueue(lits[0], cr);
    return CRef_Undef;
}

CRef GpuHelpedSolver::addLearnedClause(MinHArr<Lit>& lits) {
    stats[nbImported]++;
    // at this point, we could change the solver learnClause code to take a MinHArr instead, that would make it a bit faster
    tempLits.clear(false);
    for (int i = 0; i < lits.size(); i++) {
        tempLits.push(lits[i]);
    }
    return learnClause(tempLits, true, tempLits.size());
}

lbool GpuHelpedSolver::solve() {
    model.clear();
    conflict.clear();

    // Search:
    int curr_restarts = 0;
    while (status == l_Undef && !finisher.hasCanceledOrFinished() && withinBudget()) {
        status = search(
                luby_restart ?
                        luby(restart_inc, curr_restarts) * luby_restart_factor :
                        0); // the parameter is useless in glucose, kept to allow modifications
        curr_restarts++;
    }
    if (status == l_True) {
        // Extend & copy model:
        model.growTo(nVars());
        for (int i = 0; i < nVars(); i++)
            model[i] = value(i);
        extendModel();
        SyncOut so;
        printf("c decision level when solution found: %d\n", decisionLevel());
    }
    finisher.iveFinished(cpuThreadId);
    return status;
}

void GpuHelpedSolver::maybeReduceDB() {
    if (needToReduceCpuMemoryUsage) {
        incReduceDB = 0;
        specialIncReduceDB = 0;
        needToReduceCpuMemoryUsage = false;
    }
    Solver::maybeReduceDB();
}

lbool GpuHelpedSolver::getStatus() {
    return status;
}

GpuHelpedSolverOptions::GpuHelpedSolverOptions():
    import("GPU", "import", "import clauses from the gpu", true) {
}

GpuHelpedSolverParams GpuHelpedSolverOptions::toParams() {
    return GpuHelpedSolverParams {import};
}

void GpuHelpedSolver::printStats() {
    JObj jo;
    Solver::printStats();
    writeAsJson("assigCountSentToGpu", nextAssigPos);
}

} /* namespace Glucose */
