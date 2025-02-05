#ifndef GpuClauseSharerImpl_h
#define GpuClauseSharerImpl_h

#include "GpuClauseSharer.h"
#include "BaseTypes.cuh"
#include "GpuUtils.cuh"
#include <vector>
#include <memory>


namespace GpuShare {

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

    std::vector<unsigned long> globalStats;
    std::vector<std::vector<unsigned long>> oneSolverStats;

    std::vector<const char*> globalStatNames;
    std::vector<const char*> oneSolverStatNames;

    std::vector<std::vector<Lit>> toUnset;
    GpuClauseSharerOptions opts;

    StreamPointer sp;
    int varCount;

    Logger logger;

    void unsetPendingLocked(int threadId);


    public:
    GpuClauseSharerImpl(GpuClauseSharerOptions opts, std::function<void (const std::string &str)> logFunc);

    void setVarCount(int newCount);

    void gpuRun();

    void reduceDb();

    long getAddedClauseCount();

    long getAddedClauseCountAtLastReduceDb();

    bool hasRunOutOfGpuMemoryOnce();

    void setCpuSolverCount(int solverCount);

    long addClause(int solverId, int *lits, int count);

    bool trySetSolverValues(int threadId, int *lits, int count);

    void unsetSolverValues(int threadId, int *lits, int count);

    long trySendAssignment(int threadId);

    bool popReportedClause(int solverId, int* &lits, int &count, long &gpuClauseId);

    void getGpuMemInfo(size_t &free, size_t &total);

    void writeClausesInCnf(FILE *file);

    int getGlobalStatCount(); 

    long getGlobalStat(GlobalStats stat);

    const char* getGlobalStatName(GlobalStats stat);

    int getOneSolverStatCount(); 

    long getOneSolverStat(int solverId, OneSolverStats stat);

    const char* getOneSolverStatName(OneSolverStats stat);

    void getCurrentAssignment(int solverId, uint8_t *assig);

    long getLastAssigAllReported(int cpuSolverId);

    // Without this, we get error: incomplete type is not allowed for types in unique_ptr
    ~GpuClauseSharerImpl();
};
}

#endif
