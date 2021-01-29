# GpuShareLib

This is a library for clause sharing between threads of a parallel SAT solver. It uses the GPU via CUDA.
It is sufficient, no other mechanism for clause sharing is needed.

It's namespace is GpuShare.

It requires a few changes to make to a SAT solver to call this library. All interactions to the GPU can be done via the GpuClauseSharer class.

## Compilation changes
The first step is to copy paste this directory inside a SAT solver. It has no dependencies apart from CUDA (and the C++ standard library).
It will require changes to the Makefile to be allowed to compile .cu files with NVCC, and use NVCC for linking.

## Initialization

You will need to include GpuClauseSharer:
```
#include "gpuShareLib/GpuClauseSharer.h"
```

And create an instance of GpuClauseSharer:
```
GpuShare::GpuClauseSharerOptions csOpts;
GpuShare::GpuClauseSharer* gpuClauseSharer = GpuShare::makeGpuClauseSharerPtr(csOpts);
```

You might want to wrap it in a ```std::unique_ptr<GpuClauseSharer>```.
This instance will need to be passed to all the threads


## Continuously running the GPU
A single thread will need continously run the GPU. While it can occasionally print statistics of check if a solution has been found, it should spend most of its time running the GPU.
It should also periodically reduce the gpu clause database to prevent it from becoming too big:

```
while (<<<no solution has been found>>>) {
    gpuClauseSharer.gpuRun();
	if (gpuClauseSharer.getAddedClauseCount() - gpuClauseSharer.getAddedClauseCountAtLastReduceDb() >= gpuReduceDbPeriod) {
        gpuClauseSharer.reduceDb();
        if (!gpuClauseSharer.hasRunOutOfGpuMemoryOnce()) {
            gpuReduceDbPeriod += gpuReduceDbPeriodInc;
        }
    }
    <<< maybe print stats>>>
}
```
A good initial value for gpuReduceDbPeriod is 800000 and a good value for gpuReduceDbPeriodInc is 30000

Each solver thread will need to know its id, we will use a variable cpuThreadId for this in this example.

## Change to the CPU solver threads

### Sending clauses to the GPU
Each thread should send all the clauses it learns to the GPU, with the GpuClauseSharer method: ```long addClause(int *lits, int count);```
The representation for the literals is: 2 * var for positive literals, 2 * var + 1 for negative ones, like minisat.
Assuming that your solver uses the same representation, you can call: ``` gpuClauseSharer.addClause(&cl[0], cl.size())```

This method returns a unique long identifying the clause. You do not need to use it, but it might be useful for debugging.

### Sending assignments to the GPU
As a note, what we call 'assignment' is sometimes called 'partial assignment', some variables are undefined.

Sending assignments is needed so that the GPU can tell which clauses would have been useful for a given thread.
GpuClauseSharer maintains a representation of an assignment for every CPU thread.

The three GpuClauseSharer methods that will need to be called are:
- `bool trySetSolverValues(int cpuSolverId, int *lits, int count)` 
- `void unsetSolverValues(int cpuSolverId, int *lits, int count)`
- `long trySendAssignment(int cpuSolverId)`

Each thread should maintain a variable: int trailCopiedUntil, such that the representation of GpuClauseSharer of its assignment corresponds
to all the literals in the trail up to not including this one.

Whenever your solver backtracks, you should notify GpuClauseSharer of the literals that are being unset. Assuming that the solver has the same meaning for
trail_lim[level] and trail as minisat:

```
void Solver::unsetFromGpu(int level) {
    if (trailCopiedUntil > trail_lim[level]) {
        gpuClauseSharer.unsetSolverValues(cpuThreadId, (int*)&trail[trail_lim[level]], trailCopiedUntil - trail_lim[level]);
        trailCopiedUntil = trail_lim[level];
    }
}

void Solver::cancelUntil(int level) {
    ...
    unsetFromGpu(level);
}
```

You should have a method to try to send the assignment at a given level to the GPU:

```
void trySendAssignmentToGpu(int level) {
    int sendUntil;
    if (level < decisionLevel()) {
        unsetFromGpu(level)
        sendUntil = trail_lim[level];
    } else {
        sendUntil = trail.size()
    }
    if (trailCopiedUntil >= sendUntil) return;
    // gpuClauseSharer might already have too many assignments from our solver in which case we might not be able to pass a new assignment 
    bool success = gpuClauseSharer.trySetSolverValues(cpuThreadId, (int*)&trail[trailCopiedUntil], sendUntil - trailCopiedUntil);
    if (success) {
        trailCopiedUntil = sendUntil;
        gpuClauseSharer.trySendAssignment(cpuThreadId);
    }
}
```

Whenever a conflict is found, you can call: ```trySendAssignmentToGpu(decisionLevel() - 1)```


### Importing clauses from the GPU
The GpuClauseSharer method to call is: ``` bool popReportedClause(int cpuSolverId, int* &lits, int &count, long &gpuClauseId) ```

Instead of just importing clauses at level 0, it's essential to be able to import clauses at any time:

- If at least 2 literals of the imported clause are undef or true, the clause can directly be attached.
- If all literals are false except for one which is undef: we take the highest level of the false literals, backtrack to that level, and imply the undef literal there.
- If all literals are false except for one which is true: we take the highest level of the false literals, call it l. 
If the level of the true literal is strictly higher than l, we backtrack to l and imply it there. Otherwise, we do nothing
- If all literals are false: let’s call l1 and l2 the highest levels of literals in C with l1 <= l2.
if l1 < l2, we backtrack to l1 and imply the last literal there. Otherwise, we do conflict analysis at l1.

Importing clauses should be done regularly, for example after each decision.

You can inspire yourselves from the gpuImportClauses method of [GpuHelpedSolver](../gpu/GpuHelpedSolver.cc) for this.

