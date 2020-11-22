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
#include "ClauseUpdates.cuh"
namespace Glucose {

DClauseUpdates ClUpdateSet::getDClauseUpdates() {
    return DClauseUpdates { updates.getDArr(), vals.getDArr() };
}

HClauseUpdate::HClauseUpdate(int _clSize, int _updatePosStart, ClMetadata _clMetadata) {
    clSize = _clSize;
    updatePosStart = _updatePosStart;
    ASSERT_OP(clSize, <=, MAX_CL_SIZE);
    clMetadata = _clMetadata;
}

MinHArr<Lit> HClauseUpdates::getLits(int p) {
    HClauseUpdate up = updates[p];
    return vals.getSubArr<Lit>(up.updatePosStart, up.clSize);
}

HClauseUpdates::HClauseUpdates():
    updates(false, true),
    vals(0, true) { 
}

void HClauseUpdates::clear() {
    updates.clear(false);
    vals.resize(0, false);
}

void HClauseUpdates::addNewClause(vec<Lit> &cl, ClMetadata clMetadata) {
    HClauseUpdate clUpdate(cl.size(), vals.size(), clMetadata);
    // not using .add for because it needs constrDestr which isn't set
    int initValsSize = vals.size();
    vals.resize(initValsSize + cl.size(), false);
    for (int i = 0; i < cl.size(); i++) {
        vals[initValsSize + i] = cl[i];
    }
    updates.add(clUpdate);
}

}
