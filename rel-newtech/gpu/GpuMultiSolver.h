/***************************************************************************************

MapleGpuShare, based on MapleLCMDistChronoBT-DL -- Copyright (c) 2020, Nicolas Prevot. Uses the GPU for clause sharing.

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

#include "../mtl/Vec.h"
#include "../core/SolverTypes.h"
#include <functional>
#include "gpuShareLib/Utils.h"
#include "../simp/SimpSolver.h"

namespace GpuShare {
    class GpuClauseSharer;
}

namespace Minisat {
struct Finisher;


class PeriodicRunner;
class JsonWriter;

class GpuMultiSolver {
private:
    std::mutex solversMutex;
    vec<SimpSolver*> solvers;
    GpuShare::GpuClauseSharer &gpuClauseSharer;
    int cpuSolverCount;
    Verbosity verb;
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
    std::function<SimpSolver* (int threadId) > solverFactory;
    JsonWriter *writer;
    const GpuShare::Logger &logger;


public:
    GpuMultiSolver(Finisher &finisher, GpuShare::GpuClauseSharer &gpuClauseSharer,
            std::function<SimpSolver* (int threadId) > solverFactory, int writeClausesPeriodSec,
            Verbosity verb, double maxMemory, int gpuReduceDbPeriod, int gpuReduceDbPeriodInc, 
            JsonWriter *writer, const GpuShare::Logger &logger);

    // Meant to be called before we created more solvers
    int nVars() { return solvers[0]->nVars(); }
    void newVar() { solvers[0]->newVar(); }
    void addClause_(vec<Lit>& lits);
    void addClause(const vec<Lit>& lits);
    lbool solve(int _cpuThreadCount);
    vec<lbool>& getModel();
    void printStats();
    lbool simplify();
    void setVerbosity(Verbosity v) { verb = v; }
    Verbosity getVerbosity() { return verb; }
    double actualCpuMemUsed();
    void printGlobalStats(double cpuTime);
    void writeClausesInCnf();
    void setInitMemUsed(double mem) {initMemUsed = mem; }
    

    ~GpuMultiSolver();
};

}

#endif
