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
 * GpuHelpedSolver.h
 *
 *  Created on: 15 Jun 2017
 *      Author: nicolas
 */

#ifndef GPUHELPEDSOLVER_H_
#define GPUHELPEDSOLVER_H_

#include "simp/SimpSolver.h"
#include "utils/ConcurrentQueue.h"
#include "gpuShareLib/GpuClauseSharer.h"

#include <set>

// #define CHECK_ASSIG_ON_GPU_IS_RIGHT

namespace Glucose {

class Reported;
class OneSolverAssigs;
class Finisher;
class HostClauses;
class ClauseBatch;

enum GpuStats{
    // unfortunately this one has to be here
    // we we want to set it to the core solver stats count, so there's no
    // overlap
#define X(v) 1 +
    nbexportedgpu=(
#include "core/CoreSolverStats.h"
    0
    ),
#undef X
#define X(v) v,
#include "gpu/GpuSolverStats.h"
#undef X
} ;

struct GpuHelpedSolverParams {
    bool import;
};

class GpuHelpedSolverOptions {
public:
    BoolOption import;
    GpuHelpedSolverOptions();
    GpuHelpedSolverParams toParams();
};

struct MinClause {
    Lit *lits;
    int count;
};

class GpuHelpedSolver : public SimpSolver {
private:

    // the assig pos for the next conflict
    GpuHelpedSolverParams params;

    // it should stop increase memory usage
    volatile bool needToReduceCpuMemoryUsage;

    // It could be in a method but having it here avoids reallocating at each run
    vec<Lit> tempLits;
#ifdef CHECK_ASSIG_ON_GPU_IS_RIGHT
    vec<lbool> tempAssig;
#endif
    lbool status;
    int changedCount;
    bool quickProf;

    GpuClauseSharer &gpuClauseSharer;

    // The values of variables for GpuClauseSharer are those that we have on the trail until this
    int trailCopiedUntil;

    // Unsets all variables from the trail copied after this level
    void unsetFromTrailForGpu(int level);

    void reorderLitsAndFindLevel(vec<Lit>& lits, int& minLargestLevel, bool &hasConflict);
    void findLargestLevel(MinClause cl, int start);
    void handleReportedClause(MinClause cl, CRef &conflict, int &decisionLevelAtConflict, bool &foundEmptyClause);
    CRef insertAndLearnClause(MinClause cl, bool &foundEmptyClause);
    CRef addLearnedClause(MinClause cl);
    void foundConflictInner(int decisionLevel, vec<Lit> &learned, int lbd);
    int copyAssigsForGpuInner(int level); // returns assig id with these, or -1 if coudn't copy
    void insertStatNames();

public:
    GpuHelpedSolver(Finisher &_finisher, int cpuThreadId, GpuHelpedSolverParams params, GpuClauseSharer &_gpuClauseSharer, bool quickProf);
    GpuHelpedSolver(const GpuHelpedSolver &other, int cpuThreadId);
    void foundConflict(vec<Lit> &learned, int lbd);
    // send the state at level to the gpu
    // returns if succeedeed
    // If succeeds, exactly variables until this level will be copied for the gpu
    bool tryCopyTrailForGpu(int level);

    void tryReduceCpuMemoryUsage() { needToReduceCpuMemoryUsage = true; }

    // This will check for clauses reported by the gpu that imply something / are in conflict
    // note: this may lead to cancelling some literals
    // returns a clause found to be in conflict
    CRef gpuImportClauses(bool &foundEmptyClause);
    lbool solve();
    lbool getStatus();

    void cancelUntil(int level);

    void maybeReduceDB();

    void sendClauseToGpu(vec<Lit> &lits, int lbd);

    int getCpuThreadId() { return cpuThreadId; }
    void printStats();
};

} /* namespace Glucose */

#endif /* GPUHELPEDSOLVER_H_ */
