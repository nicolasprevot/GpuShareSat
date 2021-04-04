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
#ifndef CLAUSES_UPDATES_H_
#define CLAUSES_UPDATES_H_

#include "BaseTypes.cuh"
#include "CorrespArr.cuh"
#include "ContigCopy.cuh"

namespace GpuShare {

struct ClMetadata {
    int lbd;
    GpuClauseId gpuClauseId;
    float activity;
};


struct DClauseUpdate {
    int clSize;
    int updatePosStart;
    int clIdInSize;
};

// This class is used to add some new clauses on dClauses
// The point is that it allows to copy memory in a single chunk to the device, and then use gpu thread to update dClauses
struct DClauseUpdates {
    DArr<DClauseUpdate> updates;
    DArr<Lit> vals;
};

struct HClauseUpdate {
    int clSize;
    int updatePosStart;
    ClMetadata clMetadata;

    HClauseUpdate(int _clSize, int _updatePosStart, ClMetadata _clMetadata);
};

// We need this intermediate class between HClauseUpdates and DClauseUpdates because:
// at the time we allocate data to the contig copier, we don't know the device adresses yet, 
// since they could change if the contig copier gets resized
struct ClUpdateSet {
    ArrPair<DClauseUpdate> updates;
    ArrPair<Lit> vals;

    DClauseUpdates getDClauseUpdates();
};

// manipulated on the host to prepare the DClauseUpdates
class HClauseUpdates {
private:
    HArr<HClauseUpdate> updates;
    HArr<Lit> vals;

public:
    HClauseUpdates(const Logger &logger);
    void addNewClause(MinHArr<Lit> &lits, ClMetadata clMetadata);
    HClauseUpdate getUpdate(int p) { return updates[p]; }
    MinHArr<Lit> getLits(int p);
    MinHArr<Lit> getAllVals() { return vals; }
    int getUpdatesCount() { return updates.size(); }
    void clear();
};

}

#endif
