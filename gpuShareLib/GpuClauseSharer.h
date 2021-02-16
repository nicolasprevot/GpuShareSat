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
#ifndef GpuClauseSharer_h
#define GpuClauseSharer_h

namespace GpuShare {

struct GpuClauseSharerOptions {
    // A guideline (might not be exactly respected) of the number of blocks to use on the GPU. -1 to infer it from the machine specs.
    int gpuBlockCountGuideline;
    // A guideline (might not be exactly respected) of the number of threads per block to use on the GPU. -1 for the default.
    int gpuThreadsPerBlockGuideline;
    // Make sure that each GPU run lasts at least this time, in microseconds. This is useful when there are very few clauses on the GPU
    // since a GPU run would last a very short time, but it would negatively impact CPU performance. -1 for the default
    int minGpuLatencyMicros;
    // 0 for no verbosity, 1 for some verbosity
    int verbosity;
    // How quicly the activity of a clause decreases. The activity is bumped whenever a clause is reported. Set to a value strictly higher
    // than 0 and strictly smaller than one, or a negative value for the default.
    double clauseActivityDecay;
    // If true, will measure how much time some operations take and compute statistics for them.
    bool quickProf;

    // Set to -1 for the default. This is the maximum number of clauses reported by the GPU for each category at the start.
    int initReportCountPerCategory;

    // Page locked memory allows faster transfers between the host and the device (GPU). This is the maximum amount of memory in bytes to 
    // page lock in one allocation. If we try to allocate more than that, the memory will not be paged locked. Set to -1 to try to infer
    // it from the machine spec.
    int maxPageLockedMemory;

    GpuClauseSharerOptions() {
        gpuBlockCountGuideline = -1;
        gpuThreadsPerBlockGuideline = -1;
        minGpuLatencyMicros = -1;
        verbosity = 1;
        clauseActivityDecay = -1;
        quickProf = true;
        initReportCountPerCategory = -1;
        maxPageLockedMemory = -1;
    }
};

enum GlobalStats {
#define X(v) v,
#include "GlobalStats.h"
#undef X
};

enum OneSolverStats {
#define X(v) v,
#include "OneSolverStats.h"
#undef X
};


class GpuClauseSharer {

    public:

    /* These methods have to be always be called from the same thread, or with proper locking */

    // Do one GPU run. This method is meant to be called repeatedly by a thread, this thread spending
    // most of this time just calling this method.
    // Once this method completes, it is not guaranteed that all clauses for all assignments ready have been reported.
    virtual void gpuRun() {}


    // Delete half of the GPU clauses (those with the lowest activity).
    virtual void reduceDb() {}

    // Number of clauses that have been added, whether or not they have been deleted
    virtual long getAddedClauseCount() {return 0;}

    virtual long getAddedClauseCountAtLastReduceDb() {return 0;}


    virtual bool hasRunOutOfGpuMemoryOnce() {return false;}

    virtual void getGpuMemInfo(size_t &free, size_t &total) {}

    virtual int getGlobalStatCount() {return 0;}

    virtual long getGlobalStat(GlobalStats stat) {return 0;}

    virtual void writeClausesInCnf(FILE *file) {}

    virtual void setVarCount(int newCount) {}

    // solverId is the solver that the clause comes from. Passing this avoid reporting the clause if it triggered on previous
    // assignments sent by that solver. Pass -1 if the clause does not come from a solver
    virtual long addClause(int solverId, int *lits, int count) {return -1; }

    /* not thread safe with any other method in this class */
    virtual void setCpuSolverCount(int count) {}


    /* Thread safe methods */
    // Add a clause to the GPU. Calling this method will NOT free the lits pointer.

    virtual const char* getOneSolverStatName(OneSolverStats stat) {return NULL; }

    virtual const char* getGlobalStatName(GlobalStats stat) {return NULL; }

    virtual int getOneSolverStatCount() { return 0;}

    /* Invocations of these methods for a given solverId have to always be done from the same thread, or with proper locking */

    // Attempts to add the passed literals to the assignment of the given thread.
    // Calling this method will NOT free the lits pointer. This method is thread safe.
    // returns if succeeded. It is atomic in that either all will have been set, or none.
    virtual bool trySetSolverValues(int cpuSolverId, int *lits, int count) {return false; }

    // Unset the passed literals from the assignment of the given thread.
    // Calling this method will NOT free the lits pointer. This method is thread safe.
    // Threads are meant to call this method whenever they unset from their trail
    virtual void unsetSolverValues(int cpuSolverId, int *lits, int count) { }

    // Attempts to send the assignment for this thread to the GPU. All the clauses will be tested against it,
    // and those that trigger will be reported.
    // Returns -1 if we failed the send the current assignment, an identifier of this assignment otherwise, which always increases
    virtual long trySendAssignment(int cpuSolverId) {return -1; }

    // Returns if there was a clause reported to the given solver id. You should not free lits.
    // Calling this method will invalidate the previously returned lits pointers for this solver id.
    // The same clause maybe trigger on successive assignments from the same solver.
    // As long as a thread acts upon the reported clauses, and the clause does not trigger on future assignment,
    // the same clause will not be reported twice to the same thread. If a thread removes a reported clause as part of its
    // clause deletion policy, and this clause triggers again on an assignment from this thread, then it will
    // be reported again.
    virtual bool popReportedClause(int cpuSolverId, int* &lits, int &count, long &gpuClauseId) {return false; }

    // Returns the latest assignment for which all clauses have been reported
    virtual long getLastAssigAllReported(int cpuSolverId) { return 0;}

    // Gets the current assignment of the given cpu solver. This method is mostly intended for debugging and making sure that the GPU
    // representation of a solver assignment is the right one. The values for assig are: 0 -> true, 1 -> false, 2 -> undef
    virtual void getCurrentAssignment(int cpuSolverId, uint8_t* assig) { }

    /* Invocation of these methods can be done from any thread, but the result may not be completely up to date if is is not done
       from the thread which called the method for this cpu solver id */
    virtual long getOneSolverStat(int cpuSolverId, OneSolverStats stat) { return 0; }

    virtual ~GpuClauseSharer() { }

};

GpuClauseSharer* makeGpuClauseSharerPtr(GpuClauseSharerOptions opts);
}

#endif
