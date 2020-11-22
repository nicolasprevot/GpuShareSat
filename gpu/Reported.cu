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

#include <cuda.h>
#include "Helper.cuh"
#include "gpu/Reported.cuh"
#include "gpu/Reporter.cuh"
#include "gpu/Clauses.cuh"
#include "utils/ConcurrentQueue.h"

// #define PRINT_DETAILS_CLAUSES

namespace Glucose {

ClauseBatch::ClauseBatch():
        nextClauseToPop(0) {

}

void ClauseBatch::clear() {
    clauseDatas.clear(false);
    lits.clear(false);
    nextClauseToPop = 0;
    hadSomeReported = 0;
}

void ClauseBatch::addClause(GpuClauseId clId) {
    ClauseData clData {clId, lits.size()};
    clauseDatas.push(clData);
}

void ClauseBatch::addLit(Lit lit) {
    lits.push(lit);
}

bool ClauseBatch::popClause(MinHArr<Lit> &clLits, GpuClauseId &gpuClauseId) {
    if (nextClauseToPop >= clauseDatas.size()) return false;
    ClauseData &clData = clauseDatas[nextClauseToPop];
    int nextStart = (nextClauseToPop == clauseDatas.size() - 1) ? lits.size() : clauseDatas[nextClauseToPop + 1].posInLits;
    clLits.setSize(nextStart - clData.posInLits);
    clLits.setPtr(&(lits[clData.posInLits]));
    gpuClauseId = clData.gpuClauseId;
    nextClauseToPop++;
    return true;
}

const vec<ClauseData>& ClauseBatch::getClauseDatas() {
    return clauseDatas;
}

// maxAssigsPerSolver: the maximum number of simultaneous assignments for one solver
Reported::Reported(HostClauses &_hostClauses) :
        timesReported(0),
        totalReported(0),
        hostClauses(_hostClauses) {
}

void Reported::setSolverCount(int solverCount) {
    repClauses.growTo(solverCount);
    for (int s = 0; s < solverCount; s++) {
        // There can be at most 3 sets of 32 assignments in flight for a given solver
        repClauses[s] = std::make_unique<ConcurrentQueue<ClauseBatch>>(3);
    }
}

ClauseBatch& Reported::getClauseBatch(vec<ClauseBatch*> &perSolverBatches, int solverId) {
    if (perSolverBatches[solverId] == NULL) {
        perSolverBatches[solverId] = &(repClauses[solverId]->getNew());
        perSolverBatches[solverId]->clear();
    }
    return *perSolverBatches[solverId];
}

void Reported::fill(vec<AssigIdsPerSolver> &solvAssigIds, vec<ReportedClause> &wrongClauses) {

    // the batches for this run. Not all solvers have assignments for this run
    // We're going to set a ClauseBatch for the solvers which have one, whether some clauses were reported for them or not
    vec<ClauseBatch*> perSolverBatches(repClauses.size(), NULL);

    // This doesn't look at the wrong (reported) clauses at all
    for (int s = 0; s < solvAssigIds.size(); s++) {
        AssigIdsPerSolver &sais = solvAssigIds[s];
        if (sais.assigCount > 0) {
            ClauseBatch &clBatch = getClauseBatch(perSolverBatches, s);
            clBatch.assigIds = sais;
        }
    }

    // fill the rep clauses for all the solvers
    for (int i = 0; i < wrongClauses.size(); i++) {
        ReportedClause &wc = wrongClauses[i];
        addClause(getClauseBatch(perSolverBatches, wc.solverId), wc);
    }

    // update the ConcurrentQueues so that the solvers see the new values
    for (int s = 0; s < repClauses.size(); s++) {
        if (perSolverBatches[s] != NULL) {
            repClauses[s]->addNew();
        }
    }
    timesReported ++;
}

void Reported::addClause(ClauseBatch &clauseBatch, ReportedClause wc) {
    int gpuClId;
    // The lits are copied twice here
    // HostClauses has a weird representation of lits, not contiguous in memory
    // HostClauses could return an object that can be queried to get all the lits, would make it faster
    // but does it matter?
    hostClauses.getClause(tempLits, gpuClId, wc.gpuCref);
#ifdef PRINT_DETAILS_CLAUSES
    printf("reported clause: ");
    printV(tempLits);
    printf("\n");
#endif
    clauseBatch.addClause(gpuClId);
    for (int i = 0; i < tempLits.size(); i++) {
        clauseBatch.addLit(tempLits[i]);
    }
    clauseBatch.hadSomeReported |= wc.reportedAssignments;
    totalReported++;
}

bool Reported::getIncrReportedClauses(int solvId, ClauseBatch*& clBatch) {
    return repClauses[solvId]->getIncrInter(clBatch);
}

bool Reported::getOldestClauses(int solvId, ClauseBatch*& clBatch) {
    return repClauses[solvId]->getMin(clBatch);
}

void Reported::removeOldestClauses(int solvId) {
    repClauses[solvId]->removeMin();
}


void Reported::printStats() {
    writeAsJson("times_reported_gpu", timesReported);
}

Reported::~Reported() {

}

}
