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

namespace Glucose {

struct GpuClauseSharerOptions {
    // A guideline (might not be exactly respected) of the number of blocks to use on the GPU. -1 to infer it from the machine specs.
    int gpuBlockCountGuideline;
    // A guideline (might not be exactly respected) of the number of threads per block to use on the GPU. -1 for the default.
    int gpuThreadsPerBlockGuideline;
    // Make sure that each GPU run lasts at least this time, in microseconds. -1 for the default
    int minGpuLatencyMicros;
    // 0 for no verbosity, 1 for some verbosity
    int verbosity;
    // How quicly the activity of a clause decreases. The activity is bumped whenever a clause is reported.
    double clauseActivityDecay;
    // If true, will measure how much time some operations take.
    bool quickProf;
    
    int initReportCountPerCategory;

    GpuClauseSharerOptions() {
        gpuBlockCountGuideline = -1;
        gpuThreadsPerBlockGuideline = -1;
        minGpuLatencyMicros = -1;
        verbosity = 1;
        clauseActivityDecay = 0.99999;
        quickProf = true;
        initReportCountPerCategory = 10;
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
    virtual void gpuRun() = 0;

    // Delete half of the GPU clauses (those with the lowest activity).
    virtual void reduceDb() = 0;

    // Number of clauses that have been added, whether or not they have been deleted
    virtual long getAddedClauseCount() = 0;

    virtual long getAddedClauseCountAtLastReduceDb() = 0;


    virtual bool hasRunOutOfGpuMemoryOnce() = 0;

    virtual void getGpuMemInfo(size_t &free, size_t &total) = 0;

    virtual int getGlobalStatCount() = 0; 

    virtual long getGlobalStat(GlobalStats stat) = 0;

    virtual void writeClausesInCnf(FILE *file) = 0;


    /* not thread safe with any other method in this class */
    virtual void setCpuSolverCount(int count) = 0;


    /* Thread safe methods */
    // Add a clause to the GPU. Calling this method will NOT free the lits pointer.
    virtual long addClause(int *lits, int count) = 0;

    virtual const char* getOneSolverStatName(OneSolverStats stat) = 0;

    virtual const char* getGlobalStatName(GlobalStats stat) = 0;

    virtual int getOneSolverStatCount() = 0; 

    /* Invocations of these methods for a given solverId have to always be done from the same thread, or with proper locking */

    // Attempts to add the passed literals to the assignment of the given thread.
    // Calling this method will NOT free the lits pointer. This method is thread safe.
    // returns if succeeded. It is atomic in that either all will have been set, or none.
    virtual bool trySetSolverValues(int cpuSolverId, int *lits, int count) = 0;

    // Unset the passed literals from the assignment of the given thread.
    // Calling this method will NOT free the lits pointer. This method is thread safe.
    // Threads are meant to call this method whenever they unset from their trail
    virtual void unsetSolverValues(int cpuSolverId, int *lits, int count) = 0;

    // Attempts to send the assignment for this thread to the GPU. All the clauses will be tested against it,
    // and those that trigger will be reported.
    // This method is thread safe.
    virtual bool trySendAssignment(int cpuSolverId) = 0;

    // Returns if there was a clause reported to the given solver id. You should not free lits.
    // Calling this method will invalidate the previously returned lits pointers for this solver id.
    // The same clause maybe trigger on successive assignments from the same solver.
    // As long as a thread acts upon the reported clauses, and the clause does not trigger on future assignment,
    // the same clause will not be reported twice to the same thread. If a thread removes a reported clause as part of its
    // clause deletion policy, and this clause triggers again on an assignment from this thread, then it will
    // be reported again.
    virtual bool popReportedClause(int cpuSolverId, int* &lits, int &count, long &gpuClauseId) = 0;

    // Gets the current assignment of the given cpu solver. This method is mostly intended for debugging and making sure that the GPU
    // representation of a solver assignment is the right one. The values for assig are: 0 -> true, 1 -> false, 2 -> undef
    virtual void getCurrentAssignment(int cpuSolverId, uint8_t* assig) = 0;

    /* Invocation of these methods can be done from any thread, but the result may not be completely up to date if is is not done
       from the thread which called the method for this cpu solver id */
    virtual long getOneSolverStat(int cpuSolverId, OneSolverStats stat) = 0;



};
}

#endif
