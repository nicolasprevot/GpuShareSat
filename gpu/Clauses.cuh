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

// This file deals with the clauses that the GPU reads.
// It gets the clauses from the CPU and send them to the GPU. They are also kept on the CPU so that the CPU threads can read the reported clauses.
// It also deals with recucing the GPU clause database.

#ifndef CLAUSES_H_
#define CLAUSES_H_

#include <mutex>
#include <memory>

#include "satUtils/SolverTypes.h"
#include "mtl/Vec.h"
#include "mtl/Heap.h"
#include "gpu/CorrespArr.cuh"
#include "GpuUtils.cuh"
#include "BaseTypes.cuh"
#include "ContigCopy.cuh"
#include "ClauseUpdates.cuh"
#include "utils/Profiler.h"

#define RESCALE_CONST 1e19


namespace Glucose {



// returns the largest multiple of n which is <= a
__device__ __host__ int getLargestMod0(int a, int n);

// returns the largest value which is smaller or equ to b and equal to a modulo n
__device__ __host__ int getLargestSameMod(int a, int b, int n);

__device__ __host__ long getStartPosForClause(int clSize, int clIdInSize);

// its last lit position + 1
__device__ __host__ long getEndPosForClause(int clSize, int clIdInSize);

// Represents info for clauses with a given size
// Used to have more than vals, maybe will have more in the future?
struct DOneSizeClauses {
    DArr<Lit> vals;

    DOneSizeClauses() {
    }
};


// Contains the lits of the clauses, on the gpu
class DClauses {
private:
    // indexed by cl size
    DArr<DOneSizeClauses> perSize;
    DArr<int> limWarpPerSize;

public:
    DClauses(DArr<DOneSizeClauses> perSize, DArr<int> limWarpPerSize);

    // Deals with the values
    __device__ Lit* getStartAddrForClause(int clSize, int clIdInSize) {
        CHECK_POS(clSize, perSize);
        CHECK_POS(getStartPosForClause(clSize, clIdInSize), perSize[clSize].vals);
        return &(perSize[clSize].vals[getStartPosForClause(clSize, clIdInSize)]);
    }

    __device__ void assertInSize(int clSize, Lit *l) {
        assert(l >= &(perSize[clSize].vals[0]) && l <= &(perSize[clSize].vals[perSize[clSize].vals.size() - 1]));
    }

    __device__ int getValsSize(int clSize) { return perSize[clSize].vals.size(); }
    __device__ int getClCount(int size);

    __device__ Lit get(int clSize, int pos) {
        return perSize[clSize].vals[pos];
    }

    __device__ void set(int clSize, int pos, Lit val) {
        perSize[clSize].vals[pos] = val;
    }

    __device__ void update(int clSize, int clIdInSize, DArr<Lit> vals);

    __device__ void getClsForThread(int threadId, int &clSize, int &minClId, int &maxClId);
};

__device__ void updateClauses(DClauseUpdates clUpdates, DClauses dClauses);

struct HOneSizeClauses {
    CorrespArr<Lit> vals;
    HArr<ClMetadata> clMetadata;

    // this method makes sure that the dOneSizeClauses will have the right size
    // it doesn't copy the values though
    bool tryGetDOneSizeClauses(DOneSizeClauses &dOneSizeClauses, cudaStream_t &stream) {
        if (!vals.tryResizeDeviceToHostSize(true, &stream)) {
            return false;
        }
        dOneSizeClauses.vals = vals.getDArr();
        return true;
    }

    HOneSizeClauses(): vals(0, false), clMetadata(0, false) {
    }

    int size() {
        return clMetadata.size();
    }

    bool tryCopyAsync(cudaMemcpyKind kind, cudaStream_t &stream) {
        return vals.tryCopyAsync(kind, stream);
    }
};


// Defines clauses on the device and how to run them
class RunInfo {
    public:
    int warpCount;
    // how many warps for each cl size
    // for a size s: warp limWarpPerSize[s] is the first for which clSize is > s
    ArrPair<int> limWarpPerSize;
    ArrPair<DOneSizeClauses> dOneSizeClauses;

    bool succeeded() { return dOneSizeClauses.pointsToSomething(); }
    DClauses getDClauses() { return DClauses { dOneSizeClauses.getDArr(), limWarpPerSize.getDArr()}; }
};

void __device__ getClsForThread(int threadId, int &clSize, int &minClId, int &maxClId, DArr<int> limWarpPerSize, DClauses dClauses);

// This class keeps track of the number of clauses present for each size, their allocations,
// and their values on the host.  It also deals with keeping the device pointers and sizes (headers)
// in line with those on the host.
// It does not always update the device values in line with the host, though
// This class is not thread safe
class PerSizeKeeper{
private:
    vec<std::unique_ptr<HOneSizeClauses>> perSize;

    // This does not include clauses in clauseUpdates
    int clauseCount;
    int clauseLengthSum;

    void changeCount(int clSize, int newCount);

    void moveClause(int clSize, int oldClid, int newClId);

    float clauseActIncr;
    float clauseActDecay;

    bool tryCopyHeadersToDeviceIfNecessaryAsync(cudaStream_t &stream);
    void rescaleActivity();

public:
    PerSizeKeeper(float clauseActDecay);

    ArrPair<DOneSizeClauses> tryGetDArr(ContigCopier &cc, cudaStream_t &stream);
    // modifiers
    // If any of these is called, the previous values returned for device things getters become invalid
    // This method keeps vals and activities in line on device and host (unlike addClause)
    bool tryRemoveClausesAsync(int minLbd, int maxLimLbd, float act, cudaStream_t &stream);

    // Returns the clIdInSize, or -1 if failed to add the clause due to lack of memory
    int addClause(int clSize, MinHArr<Lit> vals, ClMetadata metadata);

    // getters for host things

    // Adds the clause and returns its clId in size
    // modifies clCountPerSize, vals, activity, metadata
    // doesn't change vals on the device
    int getClauseCount() { return clauseCount; }
    int getClauseLengthSum() { return clauseLengthSum; }
    int getClauseCount(int clSize) { return perSize[clSize]->clMetadata.size(); }
    void printStats();
    void getClause(vec<Lit> &lits, int &gpuClId, GpuCref gpuCref);
    int getLbd(int clSize, int clId) { return perSize[clSize]->clMetadata[clId].lbd; }
    float getClauseActivity(int clSize, int clId) { return perSize[clSize]->clMetadata[clId].activity;}
    void bumpClauseActivity(int clSize, int clId);

    void decayClauseAct() { clauseActIncr /= clauseActDecay; }
};


// class used on the host, but monitors the device clauses
class HostClauses {
private:
    GpuClauseId nextGpuClauseId;
    bool actOnly;
    // A guideline of how many threads should run
    int gpuThreadCountGuideline;

    HArr<int> limWarpPerSize;

    // These are the new clauses that have not been added yet
    HClauseUpdates clauseUpdates;

    std::mutex writeClauseUpdates;

    PerSizeKeeper perSizeKeeper;
    std::unique_ptr<RunInfo> runInfo;

    Profiler profiler;

    // only increases
    long clausesAddedCount;

    volatile long clausesAddedCountAtLastReduceDb;

    bool addClauseNowLocked(vec<Lit>& lits);

    int reduceDbCount;

    double getAvgActivity(int minLimLbd, int maxLimLbd);
    ArrPair<DOneSizeClauses> tryGetDClauses(ContigCopier &cc, cudaStream_t &stream);

    // howManyUnder: number of clauses with an lbd < to medLbd
    // howManyThisLbd: number of clauses with medLbd
    void getMedianLbd(int &medLbd, int &howManyUnder, int &howManyThisLbd, vec<int> &clauseCountsAtLbds);
    // it should stop increasing memory usage
    volatile bool needToReduceCpuMemoryUsage;

public:
    HostClauses(GpuDims gpuDimsGuideline, float clauseActDecay, bool actOnly);

    RunInfo makeRunInfo(cudaStream_t &stream, ContigCopier &cc);

    void decayClauseAct() {perSizeKeeper.decayClauseAct(); }

    bool needToReduceDb();
    void reduceDb(cudaStream_t &stream);

    // This method can be called by other threads.
    GpuClauseId addClause(MinHArr<Lit> clause, int lbd);

    void tryReduceCpuMemoryUsage() { needToReduceCpuMemoryUsage = true; }
    
    bool reallyNeedToCopyClausesToDevice();

    void getClause(vec<Lit>& lits, int &gpuClId, GpuCref gpuCref);

    // copies all the clauses added previously with addClause to the device
    ClUpdateSet getUpdatesForDevice(cudaStream_t &stream, ContigCopier &cc);
    void printStats();

    int getClauseCount() { return perSizeKeeper.getClauseCount(); }
    int getClauseCount(int clSize) { return perSizeKeeper.getClauseCount(clSize); }

    // Rest is visible for testing
    void fillClauseCountsAtLbds(vec<int> &vec);

    int getClauseLengthSum() { return perSizeKeeper.getClauseLengthSum(); }

    // Returns an approximation of the nth (starting from 1) lowest activity for clauses with lbd such that minLimLbd <= lbd < maxLimLbd
    // assumes that activities have been copied to the host
    // It is always striclty above
    float approxNthAct(int minLbd, int maxLbd, int target);


    void getRemovingLbdAndAct(int &minLimLbd, int &maxLimLbd, float &act, vec<int> &clauseCountsAtLbds);

    long getClausesAddedAtLastReduceDb() {return clausesAddedCountAtLastReduceDb; }
    int getReduceDbCount() {
        return reduceDbCount;
    }

    void bumpClauseActivity(GpuCref gpuCref) { perSizeKeeper.bumpClauseActivity(gpuCref.clSize, gpuCref.clIdInSize); }
    float getClauseActivity(GpuCref gpuCref) { return perSizeKeeper.getClauseActivity(gpuCref.clSize, gpuCref.clIdInSize); }

    void writeClausesInCnf(FILE *file, int varCount);

};

}

#endif /* CLAUSES_H_ */
