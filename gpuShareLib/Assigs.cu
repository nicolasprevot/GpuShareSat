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
#include "BaseTypes.cuh"
#include "Assigs.cuh"
#include "ContigCopy.cuh"
#include "my_make_unique.h"

#include <atomic>

#define PRINT_ALOT 0

namespace GpuShare {

__device__ void printV(MultiLBool multiLBool) {
    printf("tr: "); printBinary(multiLBool.isTrue); printf(" def: "); printBinary(multiLBool.isDef); printf(" "); 
}

__device__ void printV(MultiAgg multiAgg) {
    printf("t: "); printBinary(multiAgg.canBeTrue); printf(" f: "); printBinary(multiAgg.canBeFalse); printf(" u: "); printBinary(multiAgg.canBeUndef); NL;
}

void printV(VarUpdate vu) {
    printf("{\n");
    PRINT(vu.var);
    printf("newMultiLBool: ");
    printV(vu.newMultiLBool);
    printf("}\n");
}

__device__ void dSetOnMaskUint(Vals &val, Vals mask, Vals cond) {
    if (cond) {
        val = val | mask;
    }
    else {
        val = val & ~mask;
    }
}

// sets the value of current on mask to newVal
template<typename T> __device__ void atomicSetOnMaskUint(T &current, T mask, T newVal) {
    // atomicCas only takes int, so we need these conversions
    T r;
    T old;
    do {
        old = current;
        T toSet = (old & (~mask)) | (newVal & mask);
        r = atomicCAS(&current, old, toSet);
    } while(r != old);
}

__device__ void atomicSetOnMaskAgg(MultiAgg &current, Vals mask, MultiAgg newVal) {
    atomicSetOnMaskUint((ValsACas&) current.canBeTrue, (ValsACas) mask, (ValsACas) newVal.canBeTrue);
    atomicSetOnMaskUint((ValsACas&) current.canBeFalse, (ValsACas) mask, (ValsACas) newVal.canBeFalse);
    atomicSetOnMaskUint((ValsACas&) current.canBeUndef, (ValsACas) mask, (ValsACas) newVal.canBeUndef);
}

// Updates an aggregate on some bits depending on a non-aggregate
__device__ void updateAggregate(Vals aggBitsMask, Vals bitsMask, MultiLBool multiLBool, MultiAgg &multiAgg) {
    Vals bitsTrue = getTrue(multiLBool);
    Vals bitsFalse = getFalse(multiLBool);
    Vals bitsUndef = ~multiLBool.isDef;
    dSetOnMaskUint(multiAgg.canBeTrue, aggBitsMask, bitsTrue & bitsMask);
    dSetOnMaskUint(multiAgg.canBeFalse, aggBitsMask, bitsFalse & bitsMask);
    dSetOnMaskUint(multiAgg.canBeUndef, aggBitsMask, bitsUndef & bitsMask);
}

__device__ DArr<VarUpdate> assignVarUpdatesToThread(DValsPerId<VarUpdate> varUpdates, int &solverForThisThread) {
    uint threadCount = blockDim.x * gridDim.x;
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int solverCount = varUpdates.getIdCount();
    assert(solverCount <= threadCount);
    // it's rounded under, we want to have fewer threads per solver rather than more
    uint threadsPerSolver = threadCount / solverCount;
    solverForThisThread = threadId / threadsPerSolver;
    if (solverForThisThread >= varUpdates.getIdCount()) {
        solverForThisThread = -1;
        return DArr<VarUpdate>();
    }
    int firstThreadForThisSolver = solverForThisThread * threadsPerSolver;
    DArr<VarUpdate> solverVarUpdates = varUpdates.getForId(solverForThisThread);
    int minPos, maxPos;
    assignToThread(solverVarUpdates.size(), threadId - firstThreadForThisSolver, threadsPerSolver, minPos, maxPos);
    return solverVarUpdates.getSubArr<VarUpdate>(minPos, maxPos - minPos);
}

__device__ void dUpdateAssigs(DValsPerId<VarUpdate> varUpdates, DArr<DOneSolverAssigs> dOneSolverAssigs, DValsPerId<AggCorresp> dAggCorresps, DAssigAggregates aggregates) {
    int solverId;
    DArr<VarUpdate> varUpdatesThisThread = assignVarUpdatesToThread(varUpdates, solverId);
    if (solverId < 0) return;
    DArr<AggCorresp> aggCorresps = dAggCorresps.getForId(solverId);
    for (int i = 0; i < varUpdatesThisThread.size(); i++) {
        VarUpdate &vu = varUpdatesThisThread[i];
        dOneSolverAssigs[solverId].multiLBools[vu.var] = vu.newMultiLBool;
        MultiAgg newAgg { 0, 0, 0};
        Vals globAggMask = 0;
        for (int i = 0; i < aggCorresps.size(); i++) {
            updateAggregate(aggCorresps[i].aggBitMask, aggCorresps[i].bitsMask, vu.newMultiLBool, newAgg);
            globAggMask |= aggCorresps[i].aggBitMask;
        }
        atomicSetOnMaskAgg(aggregates.multiAggs[vu.var], globAggMask, newAgg);
    }
}

__host__ __device__ void setSameAsMask(Vals &u, Vals mask) {
    if ((u & mask) != 0) {
        u = ~0;
    }
    else {
        u = 0;
    }
}

__global__ void dSetAllAssigsToLast(DValsPerId<VarUpdate> varUpdates, DArr<DOneSolverAssigs> dOneSolverAssigs, DAssigAggregates dAggs) {
    int solverId;
    DArr<VarUpdate> varUpdatesThisThread = assignVarUpdatesToThread(varUpdates, solverId);
    for (int i = 0; i < varUpdatesThisThread.size(); i++) {
        VarUpdate &vu = varUpdatesThisThread[i];
        Vals lastMask = dOneSolverAssigs[solverId].lastMask;
        MultiLBool &multiLBool = dOneSolverAssigs[solverId].multiLBools[vu.var];
        setSameAsMask(multiLBool.isTrue, lastMask);
        setSameAsMask(multiLBool.isDef, lastMask);
        MultiAgg multiAgg; 
        Vals allAggBits = dOneSolverAssigs[solverId].allAggBits;
        updateAggregate(allAggBits, lastMask, multiLBool, multiAgg);
        atomicSetOnMaskAgg(dAggs.multiAggs[vu.var], allAggBits, multiAgg);
    }
}

OneSolverAssigs::OneSolverAssigs(int varCount, int &warpsPerBlock, int warpCount) :
    multiLBool(varCount),
    notCompletedMask(~0),
    // It's easier if ids don't have negative values
    currentId(0),
    firstIdUsed(0),
    varToUpdatePos(varCount, -1),
    lastVarVal(varCount, gl_Undef),
    updatesSent(0) {
        initDArr(multiLBool.getDArr(), MultiLBool {0, 0}, warpsPerBlock, warpCount);
}

Vals setOnMask(Vals x, Vals mask, bool v) {
    return x & ~mask | mask * v;
}

Vals setOnMask(Vals &x, Vals mask, bool onMask, bool notOnMask) {
    return ~mask * notOnMask | mask * onMask;
}

void OneSolverAssigs::setVarLocked(Var var, lbool val) {
    bool isSet = val != gl_Undef;
    bool isTrue = val == gl_True;
    int updatePos = varToUpdatePos[var];
    if (updatePos == -1 || updates.size() <= updatePos || updates[updatePos].var != var) {
        updates.resize(updates.size() + 1);
        VarUpdate& vu = updates[updates.size() - 1];
        // For the not completed ones: set the new values
        // For the already completed ones: set the values they had already on the gpu, which we can get from lastVarVal
        vu.newMultiLBool.isDef = setOnMask(vu.newMultiLBool.isDef, notCompletedMask, isSet, lastVarVal[var] != gl_Undef);
        vu.newMultiLBool.isTrue = setOnMask(vu.newMultiLBool.isTrue, notCompletedMask, isTrue, lastVarVal[var] == gl_True);
        vu.var = var;
        varToUpdatePos[var] = updates.size() - 1;
    }
    else {
        VarUpdate& vu = updates[updatePos];
        // set the new values for the not completed ones, don't touch the completed ones
        vu.newMultiLBool.isDef = setOnMask(vu.newMultiLBool.isDef, notCompletedMask, isSet);
        vu.newMultiLBool.isTrue = setOnMask(vu.newMultiLBool.isTrue, notCompletedMask, isTrue);
        vu.var = var;
    }
    lastVarVal[var] = val;
}

bool OneSolverAssigs::isAssignmentAvailableLocked() {
    if (currentId != firstIdUsed + assigCount()) {
        return true;
    }
    return false;
}

long OneSolverAssigs::assignmentDoneLocked() {
    int currentPos = getPos(currentId);
    assert(firstIdUsed + assigCount() != currentId);
    ASSERT_MSG(notCompletedMask & ((Vals) 1 << currentPos), PRINT(notCompletedMask); PRINT(currentPos));
    // set the bit for this pos to 0
    notCompletedMask &= ~((Vals) 1 << currentPos);
    return currentId++;
}

void OneSolverAssigs::setAggBits(int _startAggBitPos, int _endAggBitPos) {
    startAggBitPos = _startAggBitPos;
    endAggBitPos = _endAggBitPos;
}

Vals OneSolverAssigs::getMaskFromTo(int fromId, int toId) {
    Vals mask = 0;
    for (int i = fromId; i < toId; i++) {
        mask = mask | ((Vals) 1 << getPos(i));
    }
    return mask;
}

void OneSolverAssigs::setAggCorresp(AggCorresp &aggCorresp, int &aggBitPos, int &id, int bitsCount) {
    aggCorresp.aggBitMask = (Vals) 1 << aggBitPos++;
    aggCorresp.bitsMask = getMaskFromTo(id, id + bitsCount);
    id += bitsCount;
}

void OneSolverAssigs::getCurrentAssignment(uint8_t *assig) {
    memcpy(assig, &lastVarVal[0], sizeof(uint8_t) * lastVarVal.size());
}

void OneSolverAssigs::setVarCount(int varCount, cudaStream_t &stream, int &warpsPerBlock, int totalWarps) {
    std::lock_guard<std::mutex> lockGuard(lock);
    lastVarVal.insert(lastVarVal.end(), varCount - lastVarVal.size(), gl_Undef);
    varToUpdatePos.insert(varToUpdatePos.end(), varCount - varToUpdatePos.size(), -1);
    // we don't resize multiLBool here because it involves running a GPU kernel. This method can be called may times
    // we only want to run that GPU kernel once
}

void OneSolverAssigs::resizeDevVals(int varCount, int warpsPerBlock, int totalWarps, cudaStream_t &stream) {
    int oldSize = multiLBool.size();
    // There may be something using multiLBool so sync the stream using it before changing capacity
    exitIfFalse(multiLBool.tryResize(varCount, true, true, &stream), POSITION);
    initDArr(multiLBool.getDArr(), MultiLBool {0, 0}, warpsPerBlock, totalWarps, oldSize);
}

// Note: the reason why this method changes an ArrPair rather than a DArr is that if the contig copier gets resized,
// it would invalidate the DArr
DOneSolverAssigs OneSolverAssigs::copyUpdatesLocked(ArrPair<VarUpdate> &varUpdates, AssigIdsPerSolver &assigIds,
    HArr<AggCorresp> &aggCorresps) {

    assigIds.startAssigId = firstIdUsed;
    assigIds.assigCount = currentId - firstIdUsed;

    int initSize = varUpdates.size();
    varUpdates.increaseSize(initSize + updates.size());
    updatesSent += updates.size();
    if (updates.size() > 0) {
        memcpy(&(varUpdates.getHArr()[initSize]), &(updates[0]), updates.size() * sizeof(VarUpdate));
    }
    uint lastIdCopied;
    // If lastIdCopied is 0, it means nothing has been copied, we can set everything the same as 0
    if (currentId == 0) lastIdCopied = 0;
    // If currentId == firstIdUsed + assigCount() then the GPU is full
    else if (currentId == firstIdUsed + assigCount()) lastIdCopied = currentId - 1;
    // Otherwise, currentId is not completed yet, but it may have had some updates already
    else lastIdCopied = currentId;

    int bitsUsed = currentId - firstIdUsed;
    // Going to assing agg bits to non-agg bits, generally several bits for one agg bit
    // No point in using more agg bits then there are bits
    int aggBitsUsed = min(endAggBitPos - startAggBitPos, bitsUsed);

    if (aggBitsUsed != 0) {
        // some agg bits will get more bits than others
        int id = firstIdUsed;
        int aggBitPos = startAggBitPos;
        int lowBitsPerAggBit = bitsUsed / aggBitsUsed;
        int missing = bitsUsed - lowBitsPerAggBit * aggBitsUsed;
        AggCorresp aggCorresp;
        for (int i = 0; i < missing; i++) {
            setAggCorresp(aggCorresp, aggBitPos, id, lowBitsPerAggBit + 1);
            aggCorresps.add(aggCorresp);
        }
        for (int i = missing; i < aggBitsUsed; i++) {
            setAggCorresp(aggCorresp, aggBitPos, id, lowBitsPerAggBit);
            aggCorresps.add(aggCorresp);
        }
        ASSERT_OP(id, ==, currentId);
    }

    // Note: the reason for not having this method set the agg dArr here is that:
    // It's not valid if someone else later adds something else to
    // the contig copier which changes its size
    DOneSolverAssigs res;
    res.multiLBools = multiLBool.getDArr();
    res.startVals = ~notCompletedMask;
    res.lastMask = (Vals) 1 << getPos((uint) lastIdCopied);
    res.allAggBits = getMaskFromTo(startAggBitPos, endAggBitPos);
    assert(res.multiLBools.size() > 0);

    updates.clear();
    firstIdUsed = currentId;
    notCompletedMask = ~0;
    return res;
}

HostAssigs::HostAssigs(GpuDims gpuDims) :
        varCount(0),
        multiAggAlloc(0)
{
    warpsPerBlockForInit = gpuDims.threadsPerBlock / WARP_SIZE; 
    ASSERT_OP(warpsPerBlockForInit, >, 0);
    warpCountForInit = warpsPerBlockForInit * gpuDims.blockCount;
    dAssigAggregates.multiAggs = multiAggAlloc.getDArr();
    growSolverAssigs(1);
    MultiAgg multiAgg {0, ~((Vals) 0), 0};
    initDArr(multiAggAlloc.getDArr(), multiAgg, warpsPerBlockForInit, warpCountForInit);
}

OneSolverAssigs& HostAssigs::getAssigs(int solverId) { 
    return *solverAssigs[solverId];
}

template<typename T> ArrPair<T> makeArrPair(ContigCopier &cc, MinHArr<T> &vals) {
    auto res = cc.buildArrPair<T>(vals.size(), NULL);
    copy(res.getHArr(), vals);
    return res;
}

AssigsAndUpdates HostAssigs::fillAssigsAsync(ContigCopier &cc, std::vector<AssigIdsPerSolver> &assigIdsPerSolver, cudaStream_t &stream) {
    if (multiAggAlloc.size() != varCount) {
        int oldSize = multiAggAlloc.size();
        exitIfFalse(multiAggAlloc.tryResize(varCount, &stream), POSITION);
        dAssigAggregates.multiAggs = multiAggAlloc.getDArr();
        initDArr(multiAggAlloc.getDArr(), MultiAgg{0, ~((Vals) 0), 0}, warpsPerBlockForInit, warpCountForInit, oldSize);
        for (int i = 0; i < solverAssigs.size(); i++) {
            solverAssigs[i]->resizeDevVals(varCount, warpsPerBlockForInit, warpCountForInit, stream);
        }
    }

    int solverCount = solverAssigs.size();
    assigIdsPerSolver.resize(solverCount);

    ArrPair<int> solverToAggCorrespStart = cc.buildArrPair<int>(solverCount, NULL);
    HArr<AggCorresp> aggCorresps(false, false);

    ArrPair<DOneSolverAssigs> dSolverAssigs = cc.buildArrPair<DOneSolverAssigs>(solverCount, NULL);

    HValsPerId<VarUpdate> varUpdates {cc.buildArrPair<int>(solverCount, NULL), cc.buildArrPair<VarUpdate>(0, NULL)};

    for (int s = 0; s < solverCount; s++) {
        solverToAggCorrespStart.getHArr()[s] = aggCorresps.size();
        varUpdates.idToStartPos.getHArr()[s] = varUpdates.vals.getHArr().size();
        if (solverAssigs[s]->tryLock()) {
            DOneSolverAssigs dAssigs = solverAssigs[s]->copyUpdatesLocked(varUpdates.vals, 
                assigIdsPerSolver[s], aggCorresps);
            solverAssigs[s]->exitLock();
            dSolverAssigs.getHArr()[s] = dAssigs;
            assert(dAssigs.multiLBools.size() > 0);
        }
    }
    Vals aggStartVals = 0;
    for (int i = 0; i < aggCorresps.size(); i++) {
        aggStartVals |= aggCorresps[i].aggBitMask;
    }

    AssigsAndUpdates res {
        AssignmentsSet { 
            std::move(dSolverAssigs), 
            getDAssigAggregates(aggStartVals), 
            HValsPerId<AggCorresp> {std::move(solverToAggCorrespStart), makeArrPair(cc, aggCorresps)}
        },
        std::move(varUpdates)
    };
    return res;
}

void setAllAssigsToLastAsync(int warpsPerBlock, int warpCount, AssigsAndUpdates &assigsAndUpdates, cudaStream_t &stream) {
    if (assigsAndUpdates.dAssigUpdates.vals.size() != 0) {
        runGpuAdjustingDims(warpsPerBlock, warpCount, [&] (int blockCount, int threadsPerBlock) {
            dSetAllAssigsToLast<<<blockCount, threadsPerBlock, 0, stream>>>(assigsAndUpdates.dAssigUpdates.get(),
                assigsAndUpdates.assigSet.dSolverAssigs.getDArr(), assigsAndUpdates.assigSet.dAssigAggregates);
        });
    }
}

DAssigAggregates HostAssigs::getDAssigAggregates(Vals aggStartVals) {
    dAssigAggregates.startVals = aggStartVals;
    return dAssigAggregates;
}

void assignAggBitsToSolver(int &currentBit, OneSolverAssigs &solverAssig, int bitCount) {
    int start = currentBit;
    currentBit += bitCount;
    solverAssig.setAggBits(start, currentBit);
}

void HostAssigs::setVarCount(int newVarCount, cudaStream_t &stream) {
    int oldSize = multiAggAlloc.size();
    for (int i = 0; i < solverAssigs.size(); i++) {
        solverAssigs[i]->setVarCount(newVarCount, stream, warpsPerBlockForInit, warpCountForInit);
    }
    varCount = newVarCount;
}

void HostAssigs::growSolverAssigs(int solverCount) {
    int oldCount = solverAssigs.size();
    solverAssigs.resize(solverCount);
    for (int i = oldCount; i < solverCount; i++) {
        solverAssigs[i] = my_make_unique<OneSolverAssigs>(varCount, warpsPerBlockForInit, warpCountForInit);
    }

    int bitsCount = sizeof(Vals) * 8;
   
    // We need to assign some bits to the solver 
    // Some solvers may get more bits than other
    int lowBitsPerSolver = bitsCount / solverCount;
    int missing = bitsCount - lowBitsPerSolver * solverCount;
    int currentBit = 0;
    dAssigAggregates.lowBitsPerSolver = lowBitsPerSolver;
    dAssigAggregates.lowBitsStart = missing * (lowBitsPerSolver + 1);
    dAssigAggregates.lowSolvStart = missing;
    for (int i = 0; i < missing; i++) {
        assignAggBitsToSolver(currentBit, *solverAssigs[i], lowBitsPerSolver + 1);
    }
    for (int i = missing; i < solverCount; i++) {
        assignAggBitsToSolver(currentBit, *solverAssigs[i], lowBitsPerSolver);
    }
    ASSERT_OP(bitsCount, ==, currentBit);
}

void HostAssigs::printAll() {
    for (int i = 0; i < solverAssigs.size(); i++) {
        printf("solver %d:\n", i);
    }
}

}
