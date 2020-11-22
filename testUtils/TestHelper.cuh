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
#ifndef DEF_TEST_HELPER
#define DEF_TEST_HELPER

#include "gpu/GpuHelpedSolver.cuh"
#include "gpu/Assigs.cuh"
#include "gpu/GpuUtils.cuh"
#include "gpu/CompositionRoot.cuh"

// Used by tests
namespace Glucose {

class GpuRunner;

void setDefaultOptions(GpuOptions &options);

class GpuFixture {
public:
    Finisher finisher;
    vec<GpuHelpedSolver*> solvers;
    StreamPointer sp;
    int solverCount;
    CompositionRoot co;

    GpuFixture(GpuOptions options, int varCount, int solverCount, int initRepSize = 100);
    ~GpuFixture();

    void execute();
    CRef executeAndImportClauses();
    void executeAndImportClauses(vec<CRef> &res);
    void checkReportedImported(int count, int instance, bool unit);
};

void execute(GpuRunner &gpuRunner);

void addClause(HostClauses &cls, Lit lit, int lbd = 3);
void addClause(HostClauses &cls, Lit lit1, Lit lit2, int lbd = 3);
void addClause(HostClauses &cls, Lit lit1, Lit lit2, Lit lit3, int lbd = 3);
void copyToDeviceAsync(HostClauses &hCls, cudaStream_t &stream, GpuDims gpuDims);
void copyToDeviceAsync(HostClauses &hCls, cudaStream_t &stream, ContigCopier &cc, GpuDims gpuDims);

}

#endif
