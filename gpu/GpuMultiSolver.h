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
#ifndef DEF_GPU_MULTI_SOLVER
#define DEF_GPU_MULTI_SOLVER

#include "mtl/Vec.h"
#include "satUtils/SolverTypes.h"
#include <functional>
#include "utils/Utils.h"

namespace Glucose {
class GpuHelpedSolver;
class Clauses;
class Finisher;
class Reported;
class HostAssigs;
class HostClauses;
class OneSolverAssigs;
class GpuRunner;
class GpuClauseSharer;

// typedef std::function<GpuHelpedSolver* (int threadId)> SolverFactory;
//typedef int SolverFactory;

class PeriodicRunner;

class GpuMultiSolver {
private:
    std::mutex solversMutex;
    vec<GpuHelpedSolver*> helpedSolvers;
    GpuClauseSharer &gpuClauseSharer;
    int cpuSolverCount;
    std::function<GpuHelpedSolver* (int threadId)> solverFactory;
    Verbosity verb;
    float memUsedCreateOneSolver;
    std::unique_ptr<PeriodicRunner> periodicRunner;

    long gpuReduceDbPeriod;
    long gpuReduceDbPeriodInc;

    // Prints the sum of the stat among all solvers
    void printStatSum(const char* name, int stat);
    double initMemUsed;
    // we try not to use more than this amount of memory on the cpu
    // in megabytes
    double maxMemory;

    long getStatSum(int stat);
    void configure();
    bool hasTriedToLowerCpuMemoryUsage;
    Finisher &finisher;

public:
    GpuMultiSolver(Finisher &finisher, GpuClauseSharer &gpuClauseSharer,
            std::function<GpuHelpedSolver* (int threadId) > solverFactory, int varCount, int writeClausesPeriodSec,
            Verbosity verb, double initMemUsed, double maxMemory);
    void addClause_(vec<Lit>& lits);
    void addClause(const vec<Lit>& lits);
    lbool solve(int _cpuThreadCount);
    vec<lbool>& getModel();
    void printStats();
    lbool simplify();
    void setVerbosity(Verbosity v) { verb = v; }
    Verbosity getVerbosity() { return verb; }
    float getMemUsedCreateOneSolver() { return memUsedCreateOneSolver; }
    double actualCpuMemUsed();
    void printGlobalStats(double cpuTime);
    void writeClausesInCnf();
    ~GpuMultiSolver();
};

}

#endif
