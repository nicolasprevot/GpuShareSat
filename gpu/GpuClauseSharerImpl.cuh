#ifndef GpuClauseSharerImpl_h
#define GpuClauseSharerImpl_h

#include "GpuClauseSharer.h"
#include "mtl/Vec.h"
#include "satUtils/SolverTypes.h"
#include "BaseTypes.cuh"
#include "GpuUtils.cuh"
#include <memory>


namespace Glucose {

class HostAssigs;
class GpuRunner;
class Reported;
class HostClauses;

class GpuClauseSharerImpl : public GpuClauseSharer {
    protected:
    std::unique_ptr<HostAssigs> assigs;
    std::unique_ptr<GpuRunner> gpuRunner;
    std::unique_ptr<Reported> reported;
    std::unique_ptr<HostClauses> clauses;

    vec<vec<Lit>> toUnset;
    GpuClauseSharerOptions opts;

    StreamPointer sp;
    int varCount;

    void unsetPending(int threadId);

    public:
    GpuClauseSharerImpl(GpuClauseSharerOptions opts, /* TODO: we should be able to increase it */ int varCount);

    void gpuRun();

    void reduceDb();

    void setCpuSolverCount(int solverCount);

    long addClause(int *lits, int count);

    bool trySetSolverValues(int threadId, int *lits, int count);

    void unsetSolverValues(int threadId, int *lits, int count);

    bool trySendAssignment(int threadId);

    bool popReportedClause(int solverId, int* &lits, int &count, long &gpuClauseId);

    void getGpuMemInfo(size_t &free, size_t &total);

    void writeClausesInCnf(FILE *file);

};
}

#endif
