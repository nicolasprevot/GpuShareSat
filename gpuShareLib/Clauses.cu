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
 * Clauses.cpp
 *
 *  Created on: 10 Jun 2017
 *      Author: nicolas
 */

#include "Helper.cuh"
#include "Clauses.cuh"
#include <stdio.h>
#include <math.h>
#include "AssertC.cuh"
#include "Utils.h"
#include "GpuUtils.cuh"
#include "BaseTypes.cuh"
#include "my_make_unique.h"
#include "GpuClauseSharer.h"
#include <limits>

// #define LOG_MEM

namespace GpuShare {

DClauses::DClauses(DArr<DOneSizeClauses> _perSize, DArr<int> _limWarpPerSize):
    perSize(_perSize),
    limWarpPerSize(_limWarpPerSize) {
}

__device__ __host__ long getStartPosForClause(int clSize, int clIdInSize) {
    long groupId = clIdInSize / WARP_SIZE;
    int clIdInGroup = clIdInSize % WARP_SIZE;
    return groupId * WARP_SIZE * clSize + clIdInGroup;
}

__device__ __host__ long getEndPosForClause(int clSize, int clIdInSize) {
    return getStartPosForClause(clSize, clIdInSize) + (clSize - 1) * WARP_SIZE + 1;
}

__device__ __host__ long getClIdFromEndPos(int clSize, int endPos) {
    if (clSize == 0 || endPos == 0) return -1;
    int startPos = endPos - 1 - (clSize - 1) * WARP_SIZE;
    int groupId = startPos / (WARP_SIZE * clSize);
    int clIdInSize = groupId * WARP_SIZE + startPos % WARP_SIZE;
    ASSERT_OP_MSG_C(getEndPosForClause(clSize, clIdInSize), ==, endPos, PRINTCN(clSize); PRINTCN(clIdInSize));
    return clIdInSize;
}

__device__ int DClauses::getClCount(int clSize) {
    int valsSize = getValsSize(clSize);
    return getClIdFromEndPos(clSize, valsSize) + 1;
}

// returns the largest multiple of n which is <= a
__device__ __host__ int getLargestMod0(int a, int n) {
    assert(n > 0);
    assert(a >= 0);
    return (a / n) * n;
}

// returns the largest value which is smaller or equ to b and equal to a modulo n
__device__ __host__ int getLargestSameMod(int a, int b, int n) {
    return getLargestMod0(b - a, n) + a;
}

int __device__ getClSizeForWarp(int warpId, DArr<int> limWarpPerSize) {
    // this code will run for all gpu threads. But it only reads a small amount of data that all threads read
    // and all threads in a warp will always take the same branch
    // result is >= low
    int low = 0;
    // result is < high
    int high = MAX_CL_SIZE + 1;
    while (high != low + 1) {
        int candidate = (low + high) / 2 - 1;
        if (warpId < limWarpPerSize[candidate]) {
            high = candidate + 1;
        } else {
            low = candidate + 1;
        }
    }
    return low;
}

void __device__ DClauses::getClsForThread(int threadId, int &clSize, int &minClId, int &maxClId) {
    int warpId = threadId / WARP_SIZE;
    if (warpId >= limWarpPerSize[MAX_CL_SIZE]) {
        // no clauses for this warp
        clSize = 1;
        minClId = 0;
        maxClId = 0;
        return;
    }
    clSize = getClSizeForWarp(warpId, limWarpPerSize);

    int firstWarpInSize = clSize >= 1 ? limWarpPerSize[clSize - 1] : 0;
    int lastWarpInSize = limWarpPerSize[clSize];
    int groupsOf32ClausesCount = getRequired(getClCount(clSize), WARP_SIZE);
    int minGroup, maxGroup;
    // this method takes threads in argument, but pass it warps instead
    assignToThread(groupsOf32ClausesCount, warpId - firstWarpInSize, lastWarpInSize - firstWarpInSize, minGroup, maxGroup);
    minClId = minGroup * WARP_SIZE + threadId % WARP_SIZE;
    maxClId = maxGroup * WARP_SIZE + threadId % WARP_SIZE;

    if (maxClId - WARP_SIZE >= getClCount(clSize)) {
        // there are clauses for other threads in the warp but not this one
        maxClId = maxClId - WARP_SIZE;
    }
    ASSERT_OP_C(maxClId - WARP_SIZE, <=, getClCount(clSize));
}

__device__ void updateClauses(DClauseUpdates clUpdates, DClauses dClauses) {
    int min, max;
    assignToThread(clUpdates.updates.size(), min, max);

    for (int p = min; p < max; p++) {
        DClauseUpdate &up = clUpdates.updates[p];
        dClauses.update(up.clSize, up.clIdInSize, clUpdates.vals.getSubArr<Lit>(up.updatePosStart, up.clSize));
    }
}

__device__ void DClauses::update(int clSize, int clIdInSize, DArr<Lit> lits) {
    // pos in vals of DClauses
    int pos = getStartPosForClause(clSize, clIdInSize);
    for (int i = 0; i < lits.size(); i++) {
        perSize[clSize].vals[pos] = lits[i];
        pos += WARP_SIZE;
    }
}

// Things that run on the host
HostClauses::HostClauses(GpuDims gpuDimsGuideline, float _activityDecay, bool _actOnly, std::vector<unsigned long> &_globalStats) :
    nextGpuClauseId(0),
    gpuThreadCountGuideline(gpuDimsGuideline.totalCount()),
    runInfo(),
    limWarpPerSize(MAX_CL_SIZE + 1, true),
    clauseUpdates(),
    perSizeKeeper(_activityDecay, _globalStats),
    addedClauseCountAtLastReduceDb(0),
    actOnly(_actOnly),
    globalStats(_globalStats)
{
}

PerSizeKeeper::PerSizeKeeper(float _clauseActDecay, std::vector<unsigned long> &_globalStats):
        perSize(MAX_CL_SIZE + 1),
        clauseActIncr(1.0),
        clauseActDecay(_clauseActDecay),
        globalStats(_globalStats) {
    for (int clSize = 0; clSize <= MAX_CL_SIZE; clSize++) {
        perSize[clSize] = my_make_unique<HOneSizeClauses>();
    }
}

ArrPair<DOneSizeClauses> PerSizeKeeper::tryGetDArr(ContigCopier &cc, cudaStream_t &stream) {
    auto res = cc.buildArrPair<DOneSizeClauses>(MAX_CL_SIZE + 1, NULL);
    MinHArr<DOneSizeClauses> harr = res.getHArr();
    for (int clSize = 0; clSize <= MAX_CL_SIZE; clSize++) {
        if (!perSize[clSize]->tryGetDOneSizeClauses(harr[clSize], stream)) {
            // there will be some memory allocated on cc with nothing pointing on it, which is not a problem at all
            res.reset();
            return res;
        }

        ASSERT_OP_C(getClIdFromEndPos(clSize, harr[clSize].vals.size()) + 1, ==, perSize[clSize]->size()); 
    }
    return res;
}

void PerSizeKeeper::changeCount(int clSize, int newCount) {
    int endPosForClause = getEndPosForClause(clSize, newCount - 1);

    int countDiff = newCount - getClauseCount(clSize);
    globalStats[gpuClauses] += countDiff;
    globalStats[gpuClauseLengthSum] += countDiff * clSize;

    perSize[clSize]->clMetadata.resize(newCount, true);
    perSize[clSize]->vals.resize(endPosForClause, true);
}

// things that run when clauses are actually added
int PerSizeKeeper::addClause(int clSize, MinHArr<Lit> lits, ClMetadata metadata) {
    ASSERT_OP_C(clSize, >=, 1);
    int clIdInSize = getClauseCount(clSize);
    changeCount(clSize, clIdInSize + 1);
    metadata.activity = clauseActIncr;
    // pos in vals
    int pos = getStartPosForClause(clSize, clIdInSize);
    assert(perSize[clSize]);
    for (int i = 0; i < lits.size(); i++) {
        (*perSize[clSize]).vals[pos] = lits[i];
        pos += WARP_SIZE;
    }
    (*perSize[clSize]).clMetadata[clIdInSize] = metadata;
    if (metadata.activity > RESCALE_CONST) {
        rescaleActivity();
    }
    return clIdInSize;
}

void PerSizeKeeper::getClause(std::vector<Lit> &lits, int &gpuAssigId, GpuCref gpuCref) {
    lits.clear();
    ASSERT_OP_C(gpuCref.clSize, <=, MAX_CL_SIZE);
    int pos = getStartPosForClause(gpuCref.clSize, gpuCref.clIdInSize);
    for (int i = 0; i < gpuCref.clSize; i++) {
        Lit l = (*perSize[gpuCref.clSize]).vals[pos];
        lits.push_back(l);
        pos += WARP_SIZE;
    }
    gpuAssigId = perSize[gpuCref.clSize]->clMetadata[gpuCref.clIdInSize].gpuClauseId;
}

void PerSizeKeeper::bumpClauseActivity(int clSize, int clIdInSize) {
    float &act = perSize[clSize]->clMetadata[clIdInSize].activity;
    act += clauseActIncr;
    if (act > RESCALE_CONST) {
        rescaleActivity();
    }
}

void PerSizeKeeper::moveClause(int clSize, int oldClid, int newClId) {
    HOneSizeClauses& hOneSizeClauses = *perSize[clSize];
    int oldStart = getStartPosForClause(clSize, oldClid);
    int newStart = getStartPosForClause(clSize, newClId);
    for (int i = 0; i < clSize; i++) {
        hOneSizeClauses.vals[newStart + i * WARP_SIZE] = hOneSizeClauses.vals[oldStart + i * WARP_SIZE];
    }
    hOneSizeClauses.clMetadata[newClId] = hOneSizeClauses.clMetadata[oldClid];
}

bool PerSizeKeeper::tryRemoveClausesAsync(int minLimLbd, int maxLimLbd, float act, cudaStream_t &stream) {
    // The point of starting with large clause size is that we're more likely
    // to remove some so that we can free memory. We may need more memory
    // space for another clause size, which can happen due to new clauses
    // from ClauseUpdates
    std::vector<int> clauseSizesWhichFailed;
    for (int clSize = MAX_CL_SIZE; clSize >= 3; clSize--) {
        int clIdToCopyTo = 0;
        HOneSizeClauses& hOneSizeClauses = *perSize[clSize];
        HArr<ClMetadata>& metadata = hOneSizeClauses.clMetadata;
        for (int clId = 0; clId < metadata.size(); clId++) {
            ClMetadata meta = metadata[clId];
            if (meta.lbd < minLimLbd || meta.lbd < maxLimLbd && meta.activity >= act) {
                moveClause(clSize, clId, clIdToCopyTo);
                clIdToCopyTo++;
            }
        }
        changeCount(clSize, clIdToCopyTo);
        // This can fail because we may not have decreased the size and new
        // clauses may have come from the other threads
        if (!hOneSizeClauses.tryCopyAsync(cudaMemcpyHostToDevice, stream)) {
            clauseSizesWhichFailed.push_back(clSize);
        }
    }
    // try again, may succeed now thanks to memory freed from other clause
    // sizes
    for (int i = 0; i < clauseSizesWhichFailed.size(); i++) {
        HOneSizeClauses& hOneSizeClauses = *perSize[clauseSizesWhichFailed[i]];
        if (!hOneSizeClauses.tryCopyAsync(cudaMemcpyHostToDevice, stream)) {
            return false;
        }
    }
    return true;
}

void PerSizeKeeper::rescaleActivity() {
    for (int clSize = 0; clSize <= MAX_CL_SIZE; clSize++) {
        for (int clIdInSize = 0; clIdInSize < getClauseCount(clSize); clIdInSize++) {
            perSize[clSize]->clMetadata[clIdInSize].activity /= RESCALE_CONST;
        }
    }
    clauseActIncr /= RESCALE_CONST;
}

ArrPair<DOneSizeClauses> HostClauses::tryGetDClauses(ContigCopier &cc, cudaStream_t &stream) {
    return perSizeKeeper.tryGetDArr(cc, stream);
}

RunInfo HostClauses::makeRunInfo(cudaStream_t &stream, ContigCopier &cc) {
    // we want to have full warps only, so round above to the nears multiple of WARP_SIZE
    int clsCountPerThread = getRequired(globalStats[gpuClauses], gpuThreadCountGuideline);
    int clsCountPerWarp = clsCountPerThread * WARP_SIZE;
    int warpsUsedSoFar = 0;
    ArrPair<int> limWarpPerSize = cc.buildArrPair<int>(MAX_CL_SIZE + 1, NULL);
    for (int clSize = 0; clSize <= MAX_CL_SIZE; clSize++) {
        int clCounts = perSizeKeeper.getClauseCount(clSize);
        int reqWarps = getRequired(clCounts, clsCountPerWarp);
        warpsUsedSoFar += reqWarps;
        limWarpPerSize.getHArr()[clSize] = warpsUsedSoFar;
    }

    ASSERT_OP_C(globalStats[gpuClauses] == 0, ||, warpsUsedSoFar > 0);
    return RunInfo {
        warpsUsedSoFar,
        limWarpPerSize,
        tryGetDClauses(cc, stream)
    };
}

ClUpdateSet HostClauses::getUpdatesForDevice(cudaStream_t &stream, ContigCopier &cc) {
    std::lock_guard<std::mutex> lockGuard(writeClauseUpdates);
    int i = 0;
    ClUpdateSet res {
        cc.buildArrPair<DClauseUpdate>(clauseUpdates.getUpdatesCount(), NULL), 
        cc.buildArrPair<Lit>(clauseUpdates.getAllVals().size(), NULL)
    };
    copy(res.vals.getHArr(), clauseUpdates.getAllVals());
    MinHArr<DClauseUpdate> dClUpdates = res.updates.getHArr();

    while (i < clauseUpdates.getUpdatesCount()) {
        // Reasoning for doing it here:
        // It used to be done for each gpu exec
        // but how many times the gpu ran wasn't relevant
        // decay means that future clauses / activity bumps will be more important than the
        // previous ones
        decayClauseAct();
        HClauseUpdate up = clauseUpdates.getUpdate(i);
        dClUpdates[i].clIdInSize = perSizeKeeper.addClause(up.clSize, clauseUpdates.getLits(i), up.clMetadata);
        dClUpdates[i].clSize = up.clSize;
        dClUpdates[i].updatePosStart = up.updatePosStart;
        globalStats[gpuClausesAdded]++;
        i++;
    }
    // At this point, clauseUpdatesDVals will be valid and won't change until clauseUpdates.getDValsAsync is called again
    // so we can clear clause updates and let other threads add new clauses to it
    clauseUpdates.clear();
    return res;
}

// just schedules the clause to be added
GpuClauseId HostClauses::addClause(MinHArr<Lit> clause, int lbd) {
    if (clause.size() > MAX_CL_SIZE) return -1;
    std::lock_guard<std::mutex> lockGuard(writeClauseUpdates);
    GpuClauseId clId = nextGpuClauseId++;
    ClMetadata clMetadata { lbd, clId};
    clauseUpdates.addNewClause(clause, clMetadata);
    return clId;
}

bool HostClauses::reallyNeedToCopyClausesToDevice() {
    std::lock_guard<std::mutex> lockGuard(writeClauseUpdates);
    // If we have 10 threads and each one produces 5000 clauses per second, that would make it
    // 50000 clauses per second, so this would make it 1 seconds
    return clauseUpdates.getUpdatesCount() > 50000;
}

void HostClauses::getClause(std::vector<Lit> &lits, int &gpuAssigId, GpuCref gpuCref) {
    perSizeKeeper.getClause(lits, gpuAssigId, gpuCref);
}

void HostClauses::fillClauseCountsAtLbds(std::vector<int> &vec) {
    for (int clSize = 0; clSize <= MAX_CL_SIZE; clSize++) {
        for (int clId = 0; clId < perSizeKeeper.getClauseCount(clSize); clId++) {
            vec[perSizeKeeper.getLbd(clSize, clId)] ++;
        }
    }
}

void HostClauses::getMedianLbd(int &medLbd, int &howManyUnder, int &howManyThisLbd, std::vector<int> &clauseCountsAtLbds) {
    int seen = 0;
    for (int lbd = 0; lbd <= MAX_CL_SIZE; lbd++ ) {
        seen += clauseCountsAtLbds[lbd];
        if (seen >= globalStats[gpuClauses] / 2) {
            medLbd = lbd;
            howManyUnder = seen - clauseCountsAtLbds[lbd];
            howManyThisLbd = clauseCountsAtLbds[lbd];
            return;
        }
    }
    throw;
}

void HostClauses::getRemovingLbdAndAct(int &minLimLbd, int &maxLimLbd, float &act, std::vector<int> &clauseCountsAtLbds) {
    int howManyUnder, howManyThisLbd;
    if (actOnly) {
        minLimLbd = 0;
        maxLimLbd = MAX_CL_SIZE;
        act = approxNthAct(minLimLbd, maxLimLbd, globalStats[gpuClauses] / 2);
        printf("c Keeping clauses with act >= %g\n", act);
        return;
    } else {
        getMedianLbd(minLimLbd, howManyUnder, howManyThisLbd, clauseCountsAtLbds); // Never remove clauses with lbd <= 2
        if (minLimLbd <= 2) {
            printf("Keeping all clauses with lbd <= 2, there are %d of them\n", howManyUnder + howManyThisLbd);
            minLimLbd = 2;
            maxLimLbd = 3;
            act = 0.0;
            return;
        }
        maxLimLbd = minLimLbd + 1;
    }
    printf("Keeping clauses with lbd < %d, there are %d of them\n", minLimLbd, howManyUnder);
    int target = globalStats[gpuClauses] / 2;
    assert(howManyUnder <= target);
    assert(howManyUnder + howManyThisLbd >= target);
    int howManyKeepThisLbd = target - howManyUnder;
    act = approxNthAct(minLimLbd, maxLimLbd, howManyThisLbd - howManyKeepThisLbd);
    printf("Also keeping clauses with lbd = %d and activity >= %g\n", minLimLbd, act);
}

void printMem() {
    size_t freeMem;
    size_t totalMem;
    exitIfError(cudaMemGetInfo(&freeMem, &totalMem), POSITION);
    printf("c There is %ld free memory out of %ld\n", freeMem, totalMem);
}

void HostClauses::reduceDb(cudaStream_t &stream) {
    TimeGauge tg(globalStats[timeSpentReduceGpuDb], true);
    std::vector<int> clauseCountsAtLbds(MAX_CL_SIZE + 1, 0);
    addedClauseCountAtLastReduceDb = globalStats[gpuClausesAdded];
    fillClauseCountsAtLbds(clauseCountsAtLbds);

    globalStats[gpuReduceDbs]++;
    printf("c Reducing gpu clause db, clause count is %ld\n", globalStats[gpuClauses]);
    printMem();

    int minLimLbd, maxLimLbd;
    float act;
    while(true) {
        getRemovingLbdAndAct(minLimLbd, maxLimLbd, act, clauseCountsAtLbds);
        if (perSizeKeeper.tryRemoveClausesAsync(minLimLbd, maxLimLbd, act, stream)) {
            break;
        } else {
            printf("Failed to copy clauses to the gpu after removing them due to low memory, going to remove some more\n");
        }
    }

    if (!actOnly) {
        int cls = 0;
        int totalClauseCount = globalStats[gpuClauses];
        // Although these clause counts at lbd were computed before the removeClause, they are still valid after
        for (int lbd = 0; lbd < minLimLbd; lbd++) {
            int atThisLbd = clauseCountsAtLbds[lbd];
            if (atThisLbd != 0) {
                printf("c Clause count at lbd %d : %d\n", lbd, atThisLbd);
            }
            cls += atThisLbd;
        }
        if (totalClauseCount > cls) {
            printf("c Clause count at lbd %d : %d\n", minLimLbd, totalClauseCount - cls);
        }
    }
    exitIfError(cudaStreamSynchronize(stream), POSITION);
    printf("c Done reducing gpu clause db, clause count is %ld\n", globalStats[gpuClauses]);
    printMem();
}


double HostClauses::getAvgActivity(int minLimLbd, int maxLimLbd) {
    double res = 0;
    int totalClCount = 0;
    for (int clSize = 1; clSize <= MAX_CL_SIZE; clSize++) {
        int count = perSizeKeeper.getClauseCount(clSize);
        for (int clId = 0; clId < count; clId++) {
            int lbd = perSizeKeeper.getLbd(clSize, clId);
            if (minLimLbd <= lbd && lbd < maxLimLbd) {
                res += perSizeKeeper.getClauseActivity(clSize, clId);
            }
        }
        totalClCount += count;
    }
    ASSERT_OP_C(totalClCount, >, 0);
    return res / totalClCount;
}

// We have some buckets, each of width inc
// bucket 0 is from min to min + inc, bucket 1 from min + inc to min + 2 inc...
// Which buck does value belong to ?
int getBucket(double min, double value, double inc) {
    return (int) floor((value - min) / inc);
}

float HostClauses::approxNthAct(int minLimLbd, int maxLimLbd, int n) {
    // We are going to divide the activities into buckets
    // And check the bucket that gives us the right target
    // The point is that we only need to iterate over the clauses once
    // Because activities are generally a logarithmic thing (ie they get divided by
    // a certain factor regularly), it makes sense to use a log scale
    if (n == 0) return 0.0;
    int bucketsCount = 20000;
    std::vector<int> clCounts(bucketsCount, 0);
    float lowestLog = log(std::numeric_limits<float>::min());
    float largestLog = log(std::numeric_limits<float>::max());
    float stepLog = (largestLog - lowestLog) / bucketsCount;

    for (int clSize = 1; clSize <= MAX_CL_SIZE; clSize++) {
        int count = perSizeKeeper.getClauseCount(clSize);
        for (int clId = 0; clId < count; clId++) {
            int lbd = perSizeKeeper.getLbd(clSize, clId);
            if (lbd < minLimLbd || lbd >= maxLimLbd) continue;
            int bucket = getBucket(lowestLog, log(perSizeKeeper.getClauseActivity(clSize, clId)), stepLog);
            if (bucket < 0) bucket = 0;
            if (bucket >= bucketsCount) bucket = bucketsCount - 1;
            clCounts[bucket] ++;
        }
    }
    int cls = 0;
    for (int bucket = 0; bucket < bucketsCount; bucket++) {
        cls += clCounts[bucket];
        if (cls >= n) {
            // bucket + 1 because we round above
            return exp(lowestLog + (bucket + 1) * stepLog);
        }
    }
    throw;
}

void HostClauses::writeClausesInCnf(FILE *file, int varCount) {
    printf("p cnf %d %ld\n", varCount, globalStats[gpuClauses]);
    std::vector<Lit> lits;
    int gpuAssigId;
    for (int clSize = 1; clSize <= MAX_CL_SIZE; clSize++) {
        int count = getClauseCount(clSize);
        for (int clIdInSize = 0; clIdInSize < count; clIdInSize++) {
            GpuCref gpuCref {clSize, clIdInSize};
            getClause(lits, gpuAssigId, gpuCref);
            writeClause(file, lits);
        }
    }
}

void writeClause(FILE *file, const std::vector<Lit>& lits) {
    for (int i = 0; i < lits.size(); i++) {
        int val = (var(lits[i]) + 1) * (sign(lits[i]) ? -1 : 1);
        fprintf(file, "%d ", val);
    }
    fprintf(file, "0\n");
}

}
