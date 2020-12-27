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
 * Reported.cuh
 *
 *  Created on: 2 Jun 2018
 *      Author: nicolas
 */

#ifndef REPORTED_CUH_
#define REPORTED_CUH_
#include "CorrespArr.cuh"
#include "BaseTypes.cuh"
#include "GpuUtils.cuh"
#include "Vec.h"
#include <set>

namespace GpuShare {

class HostClauses;

// This class is rather similar to ClauseUpdates except that it's on the host only while ClauseUpdates also copies things
// to the device and thus uses some CorrespArr
// We can add some clauses to it, and they can later be retrieved
// The main point of this class is that it does very few allocations, just reuses previous memory
// This class is not thread safe
struct ClauseData {
    GpuClauseId gpuClauseId;
    int posInLits;
};

class ClauseBatch {
private:
    vec<Lit> lits;
    vec<ClauseData> clauseDatas;
    int nextClauseToPop;

public:
    AssigIdsPerSolver assigIds;
    uint hadSomeReported;
    int assigWhichKnowsAboutThese;

    void clear();

    ClauseBatch();

    void addClause(GpuClauseId clId);
    // adds the lits of the previous clause
    void addLit(Lit lit);

    // once clear is called, the MinHArr won't be valid any more
    bool popClause(MinHArr<Lit> &lits, GpuClauseId &gpuClauseId);

    const vec<ClauseData>& getClauseDatas();
};

template<typename T> class ConcurrentQueue;

// This class gets the output from a gpu run and then rearranges it in a way which is efficient to query by a solver
class Reported {
private:
    vec<std::unique_ptr<ConcurrentQueue<ClauseBatch>>> repClauses; // first index: solver

    vec<vec<unsigned long>> &oneSolverStats;
    // only used from the solver threads
    vec<std::set<GpuClauseId>> clausesToNotImportAgain;
    // this is only accessed from the solver threads
    vec<ClauseBatch*> currentClauseBatches;
    vec<long> lastSentAssigId;
    vec<Lit> tempLits;
    HostClauses &hostClauses;

    vec<long> lastAssigAllReported;

    ClauseBatch& getClauseBatch(vec<ClauseBatch*> &perSolverBatches, int solverId);
    void addClause(ClauseBatch &clauseBatch, ReportedClause wc);

    // called by the solver threads
    bool getIncrReportedClauses(int solvId, ClauseBatch*& clBatch);
    bool getOldestClauses(int solvId, ClauseBatch*& clBatch);
    void removeOldestClauses(int solvId);

public:
    Reported(HostClauses &hostClauses, vec<vec<unsigned long>> &oneSolverStats);

    // This isn't known yet when the object is created which is why we have to set it later
    void setSolverCount(int solverCount);

    // solvAssigs tell us the solver id / and solverAssigId for a given position in the reported clauses
    void fill(vec<AssigIdsPerSolver> &solvAssigs, vec<ReportedClause> &wrongClauses);

    void assigWasSent(int solverId, long solverAssigId) { lastSentAssigId[solverId] = solverAssigId; }
    // called by the solver threads
    bool popReportedClause(int solverId, MinHArr<Lit> &lits, GpuClauseId &gpuClauseId);

    long getLastAssigAllReported(int cpuSolverId) {return lastAssigAllReported[cpuSolverId]; }

    ~Reported();
};

}

#endif /* REPORTER_CUH_ */
