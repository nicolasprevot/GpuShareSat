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

// This file is about getting the assignments from the CPU solvers, and rearranging them in a way which is efficient
// to check on the gpu

#ifndef Assigs_h
#define Assigs_h

#include "BaseTypes.cuh"
#include "CorrespArr.cuh"
#include "mtl/Vec.h"
#include "satUtils/SolverTypes.h"
#include "GpuUtils.cuh"
#include "ContigCopy.cuh"

#include <atomic>
#include <memory>

namespace GpuShare {

inline int assigCount() {
    return sizeof(Vals) * 8;
}
typedef uint32_t VarsVal;

__device__ void printVD(MultiLBool multiLBool);

struct MultiAgg {
    Vals canBeTrue;
    Vals canBeUndef;
    Vals canBeFalse;
};

__device__ __host__ void printV(MultiAgg multiAgg);
__device__ void printVD(MultiAgg multiAgg);

struct DAssigAggregates {
    DArr<MultiAgg> multiAggs;
    Vals startVals;
    // the solvers from lowSolvStart have that many agg bits
    // the ones before have this + 1
    Vals lowBitsPerSolver;
    Vals lowBitsStart;
    Vals lowSolvStart;

    __device__ int getSolver(int bitPos) {
        if (bitPos >= lowBitsStart) {
            return (bitPos - lowBitsStart) / lowBitsPerSolver + lowSolvStart;
        }
        return bitPos / (lowBitsPerSolver + 1);
    }

    __device__ int getEndBitPos(int solver) {
        if (solver >= lowSolvStart) {
            return (solver + 1 - lowSolvStart) * lowBitsPerSolver;
        }
        return (solver + 1) * (lowBitsPerSolver + 1);

    }
};

struct AggCorresp {
    Vals aggBitMask;
    Vals bitsMask;
};

template<typename T>
struct DValsPerId {
    DArr<int> idToStartPos;
    DArr<T> vals;

    __device__ DArr<T> getForId(int id) {
        int last = (id == idToStartPos.size() - 1) ? vals.size() : idToStartPos[id + 1];
        return vals.template getSubArr<T>(idToStartPos[id], last - idToStartPos[id]);
    }

    __device__ int getIdCount() { return idToStartPos.size(); }
};

template<typename T>
struct HValsPerId {
    ArrPair<int> idToStartPos;
    ArrPair<T> vals;

    DValsPerId<T> get() {
        return DValsPerId<T> {idToStartPos.getDArr(), vals.getDArr()};
    }
};

struct DOneSolverAssigs {
    DArr<MultiLBool> multiLBools;
    Vals startVals; // key: solver. Each val is set to 1 if it should be considered, 0 otherwise
    Vals lastMask; // only one bit set: for the last assignment
    // this doesn't change from one run to the next
    Vals allAggBits;
};

// This contains info about assigs on the device and what solver / solver assig they correspond to
struct AssignmentsSet {
    ArrPair<DOneSolverAssigs> dSolverAssigs;
    DAssigAggregates dAssigAggregates;
    HValsPerId<AggCorresp> aggCorresps;

    int getTotalAssigCount();
};

struct VarUpdate {
    int var;
    MultiLBool newMultiLBool;
};

void printV(VarUpdate vu);

__device__ void dUpdateAssigs(DValsPerId<VarUpdate> varUpdates, DArr<DOneSolverAssigs> dOneSolverAssigs, DValsPerId<AggCorresp> dAggCorresps, DAssigAggregates aggregates);

struct AssigsAndUpdates {
    AssignmentsSet assigSet;
    HValsPerId<VarUpdate> dAssigUpdates;
};

/*
// Used to copy assigs to the gpu
class VarsCopier {
    void 
};
*/

template<typename T> class ArrPair;
// This class is the link between the GpuHelpedSolver and the gpu
// In this class, assig ids only increase, assig pos are limited between 0 and 31
class OneSolverAssigs {
private:
    ArrAllocator<MultiLBool> multiLBool;
    // This is a bit weird, but it's so that a solver can acquire a lock, and then set all the modified assigs, and then free the lock 
    std::mutex lock;
    // mask for assignments which are either free or aren't completed
    Vals notCompletedMask;

    long updatesSent;

    // tells us the last value set for a var
    vec<lbool> lastVarVal;

    vec<VarUpdate> updates;

    // the update pos in updates for a var if there's already one, -1 otherwise
    // values here may not actually be right (they may be for a previous run)
    vec<int> varToUpdatePos;

    // This is the first id for which the assignment is used. Can be equal to currentId, in which case it is not
    long firstIdUsed;

    long currentId;

    void copyCompletedLocked();

    int getPos(int id) { return id % assigCount(); }

    // position of the first bit which belongs to this solver in the aggregate
    int startAggBitPos;
    int endAggBitPos;

    Vals getMaskFromTo(int fromId, int toId);
    void setAggCorresp(AggCorresp &aggCorresp, int &aggBitPos, int &id, int bitsCount);



public:
    OneSolverAssigs(int varCount, int &warpsPerBlock, int warpCount);
    void setVarLocked(Var var, lbool val);
    void enterLock() { lock.lock(); }
    bool tryLock() { return lock.try_lock(); }
    void exitLock() { lock.unlock(); }
    bool isAssignmentAvailableLocked();
    long assignmentDoneLocked();
    void getCurrentAssignment(uint8_t* assig);
    long getUpdatesSent() { return updatesSent; }
    DOneSolverAssigs copyUpdatesLocked(ArrPair<VarUpdate> &varUpdates, AssigIdsPerSolver &assigIds, HArr<AggCorresp> &aggCorresps);

    void setAggBits(int startAggBitPos, int endAggBitPos);
    bool hasSomethingToCopy() { return currentId != firstIdUsed; }
};

class HostAssigs {
private:
    int varCount;
    // The reason to have unique_ptr rather than the raw objects is:
    // OneSolverAssigs have locks as members, and they don't seem to enjoy
    // being realloced directly, which vec does
    vec<std::unique_ptr<OneSolverAssigs>> solverAssigs;

    ArrAllocator<MultiAgg> multiAggAlloc;
    DAssigAggregates dAssigAggregates;

    int solverCount() { return solverAssigs.size(); }

    DAssigAggregates getDAssigAggregates(Vals aggStartVals);
    int warpsPerBlockForInit;
    int warpCountForInit;

public:
    HostAssigs(int varCount, GpuDims gpuDims);
    long getChangeCount();

    void growSolverAssigs(int newCount);

    int getVarCount() {return varCount;}

    OneSolverAssigs& getAssigs(int solverId);

    AssigsAndUpdates fillAssigsAsync(ContigCopier &cc, vec<AssigIdsPerSolver> &assigIdsPerSolver, cudaStream_t &stream);

    void printStats();

    void printAll();
};

void setAllAssigsToLastAsync(int warpsPerBlock, int warpsCount, AssigsAndUpdates &assigsAndUpdates, cudaStream_t &stream);

}

#endif
