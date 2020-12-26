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

#include "GpuHelpedSolver.h"
#include "gpuShareLib/Utils.h"
#include "gpuShareLib/Profiler.h"

#include <chrono>
#include <thread>
#include <sstream>
#include <mutex>


#define PRINT_ALOT 0

namespace Glucose {

GpuHelpedSolver::GpuHelpedSolver(const GpuHelpedSolver &other, int _cpuThreadId) :
        SimpSolver(other, _cpuThreadId), params(other.params),
        needToReduceCpuMemoryUsage(other.needToReduceCpuMemoryUsage), 
        status(other.status), changedCount(other.changedCount), quickProf(other.quickProf),
        gpuClauseSharer(other.gpuClauseSharer), 
        trailCopiedUntil(other.trailCopiedUntil) {

}

GpuHelpedSolver::GpuHelpedSolver(Finisher &_finisher, int _cpuThreadId, GpuHelpedSolverParams _params, GpuShare::GpuClauseSharer &_gpuClauseSharer, bool _quickProf) :
        SimpSolver(_cpuThreadId, _finisher), params(_params),
        needToReduceCpuMemoryUsage(false), status(l_Undef), changedCount(0), quickProf(_quickProf),
        gpuClauseSharer(_gpuClauseSharer), trailCopiedUntil(0) {
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
    TimeGauge tg(stats[timeSpentImportingClauses], quickProf);
    foundEmptyClause = false;

    CRef confl = CRef_Undef;
    int decisionLevelAtConflict = -1;
    int *litsAsInt;
    int count;
    long gpuClauseId;
    while (gpuClauseSharer.popReportedClause(cpuThreadId, litsAsInt, count, gpuClauseId)) {
        Lit *lits = (Lit*) litsAsInt;
        if (params.import) {
            MinClause litsArr{lits, count};
            handleReportedClause(litsArr, confl, decisionLevelAtConflict, foundEmptyClause);
        }
    }

    if (decisionLevel() == decisionLevelAtConflict) {
        return confl;
    }
    return CRef_Undef;
}

void GpuHelpedSolver::sendClauseToGpu(vec<Lit> &lits, int lbd) {
    gpuClauseSharer.addClause((int*) &lits[0], lits.size());
    if (lits.size() == 1) {
        stats[nbExportedUnit]++;
    } else {
        stats[nbexportedgpu]++;
    }
}

void GpuHelpedSolver::handleReportedClause(MinClause lits, CRef &conflict, int &decisionLevelAtConflict, bool &foundEmptyClause) {
    CRef cr = insertAndLearnClause(lits, foundEmptyClause);
    if (cr != CRef_Undef) {
        decisionLevelAtConflict = decisionLevel();
        conflict = cr;
    }
}

void GpuHelpedSolver::unsetFromTrailForGpu(int level) {
    // if level is decisionLevel, there's no trail_lim for it
    if (level < decisionLevel()) {
        // doing this because trail may actually be copied until less than that
        if (trailCopiedUntil > trail_lim[level]) {
            gpuClauseSharer.unsetSolverValues(cpuThreadId, (int*)&trail[trail_lim[level]], trailCopiedUntil - trail_lim[level]);
            trailCopiedUntil = trail_lim[level];
        }
    }

}

bool GpuHelpedSolver::tryCopyTrailForGpu(int level) {
    int max;
    if (level < decisionLevel()) {
        unsetFromTrailForGpu(level);
        max = trail_lim[level];
    } else {
        max = trail.size();
    }
    bool result = true;
    if (trailCopiedUntil < max) {
        result = gpuClauseSharer.trySetSolverValues(cpuThreadId, (int*)&trail[trailCopiedUntil], max - trailCopiedUntil);
        if (result) trailCopiedUntil = max;
    }
    long assigId = 0;
    if (result) {
        assigId = gpuClauseSharer.trySendAssignment(cpuThreadId);
        result = assigId >= 0;
    }
    if (result) {
        stats[nbAssignmentsSent]++;
#ifdef CHECK_ASSIG_ON_GPU_IS_RIGHT
        tempAssig.resize(nVars());
        gpuClauseSharer.getCurrentAssignment(cpuThreadId, (uint8_t*) &tempAssig[0]);
        for (int i = 0; i < nVars(); i++) {
            lbool exp;
            if (this->level(i) <= level) exp = value(i);
            else exp = l_Undef;
            ASSERT_OP_MSG(exp, ==, tempAssig[i], PRINT(i));
        }
#endif
    }
    else stats[nbFailureFindAssignment]++;

    return result;
}

void GpuHelpedSolver::cancelUntil(int level) {
    unsetFromTrailForGpu(level);
    Solver::cancelUntil(level);
}

void GpuHelpedSolver::foundConflict(vec<Lit> &learned, int lbd) {
    foundConflictInner(decisionLevel() - 1, learned, lbd);
}

void GpuHelpedSolver::foundConflictInner(int level, vec<Lit> &learned, int lbd) {
    tryCopyTrailForGpu(level);
    if (lbd >= 0) {
        sendClauseToGpu(learned, lbd);
    }
}

// finds the lit with the largest level among lits from start to end and put it in start
void GpuHelpedSolver::findLargestLevel(MinClause cl, int start) {
    for (int i = start + 1; i < cl.count; i++) {
        if (litLevel(cl.lits[i]) > litLevel(cl.lits[start])) {
            std::swap(cl.lits[i], cl.lits[start]);
        }
    }
}

// Learns the clause, and attach it
// May lead to canceling to backtracking if this clause leads to an implication at a level strictly lower
// than the current level
// returns a non-undef cref if this clause is in conflict
CRef GpuHelpedSolver::insertAndLearnClause(MinClause cl, bool &foundEmptyClause) {
    stats[nbImported]++;
    Lit *lits = cl.lits;
    int count = cl.count;
    // if only one literal isn't false, it will be in 0 of lits
    findLargestLevel(cl, 0);
    // if only two literals aren't set, they will be in 0 and 1 of lits
    findLargestLevel(cl, 1);
    if (count == 0) {
        stats[nbImportedValid]++;
        foundEmptyClause = true;
        return CRef_Undef;
    }
    if (count == 1) {
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
        addLearnedClause(cl);
        return CRef_Undef;
    }
    if (value(lits[0]) == l_True
            && level(var(lits[0])) <= level(var(lits[1]))) {
        // We're implying a literal at a level <= to what it is already. Just learn the clause, it's not doing anything now, though
        addLearnedClause(cl);
        return CRef_Undef;
    }
    if ((value(lits[0]) == l_False)
            && (level(var(lits[1])) == level(var(lits[0])))) {
        // conflict
#ifdef DEBUG
        for (int i = 0; i < count; i++) {
            assert(value(lits[i]) == l_False);
        }
#endif
        assert(value(lits[1]) == l_False);
        stats[nbImportedValid]++;
        cancelUntil(level(var(lits[1])));
        qhead = trail.size();
        return addLearnedClause(cl);
    }
    // lit 0 is implied by the rest, may currently be undef or false
#ifdef DEBUG
    for (int i = 1; i < count; i++) {
        assert(value(lits[i]) == l_False);
    }
#endif
    CRef cr = addLearnedClause(cl);
    cancelUntil(level(var(lits[1])));
    stats[nbImportedValid]++;
    uncheckedEnqueue(lits[0], cr);
    return CRef_Undef;
}

CRef GpuHelpedSolver::addLearnedClause(MinClause cl) {
    // at this point, we could change the solver learnClause code to take a MinHArr instead, that would make it a bit faster
    tempLits.clear(false);
    for (int i = 0; i < cl.count; i++) {
        tempLits.push(cl.lits[i]);
    }
    return learnClause(tempLits, true, tempLits.size());
}

lbool GpuHelpedSolver::solve() {
    model.clear();
    conflict.clear();

    // Search:
    int curr_restarts = 0;
    while (status == l_Undef && !finisher.shouldIStop(cpuThreadId) && withinBudget()) {
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
        if (verb.global > 0)  printf("c decision level when solution found: %d\n", decisionLevel());
    }
    if (status == l_True || status == l_False) { 
        finisher.oneThreadIdWhoFoundAnAnswer = cpuThreadId;
        finisher.stopAllThreads = true;
    }
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
    for (int i = 0; i < gpuClauseSharer.getOneSolverStatCount(); i++) {
        GpuShare::OneSolverStats oss = static_cast<GpuShare::OneSolverStats>(i);
        writeAsJson(gpuClauseSharer.getOneSolverStatName(oss), gpuClauseSharer.getOneSolverStat(cpuThreadId, oss));
    }
}

} /* namespace Glucose */
