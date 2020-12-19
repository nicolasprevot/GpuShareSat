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

struct GpuClauseShareOptions {
    // Maximum number of cpu threads
    int cpuThreadCount;
    // A guideline (might not be exactly respected) of the number of blocks to use on the GPU. -1 to infer it from the machine specs.
    int gpuBlockCountGuideline;
    // A guideline (might not be exactly respected) of the number of threads per block to use on the GPU. -1 for the default.
    int gpuThreadsPerBlockGuideline;
    // Make sure that each GPU run lasts at least this time, in microseconds. -1 for the default
    int minGpuLatencyMicros;
    // 0 for no verbosity, 1 for some verbosity
    int verbosity;

    double clauseActivityDecay;
}

class GpuClauseSharer {
    private:
    std::unique_ptr<HostAssigs> assigs;
    std::unique_ptr<GpuRunner> gpuRunner;
    std::unique_ptr<Reported> reported;
    std::unique_ptr<HostClauses> clauses;

    public:
    GpuClauseSharer(GpuClauseShareOptions options);

    // Do one GPU run. This method is meant to be called repeatedly by a thread, this thread spending
    // most of this time just calling this method.
    // Once this method completes, it is not guaranteed that all clauses for all assignments ready have been reported.
    void gpuRun();

    // Delete half of the GPU clauses (those with the lowest activity).
    void reduceDb();

    // Add a clause to the GPU. Calling this method will NOT free the lits pointer. This method is thread safe.
    void addClause(int *lits, int count);

    // Adds the passed literals to the assignment of the given thread.
    // Calling this method will NOT free the lits pointer. This method is thread safe.
    void setThreadValues(int threadId, int *lits, int count);

    // Unset the passed literals from the assignment of the given thread.
    // Calling this method will NOT free the lits pointer. This method is thread safe.
    void unsetThreadValues(int threadId, int *lits, int count);

    // The assignment for this current thread will be sent to the GPU. All the clauses will be tested against it,
    // and those that trigger will be reported.
    // This method is thread safe.
    void assignmentIsReadyToTest(int threadId);

    // Returns a clause reported to the given thread id. If there is no clause to report, lits will
    // be set to NULL and count to -1. You should not free lits.
    // Calling this method will invalidate the previously returned lits pointers for this threadId.
    // This method may report the same clause to the same thread several times, but it avoids doing it too many times.
    void getReportedClause(int threadId, int &*lits, int &count);
};


#endif
