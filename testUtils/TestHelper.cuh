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

#include "gpuShareLib/GpuClauseSharerImpl.cuh"
#include "gpuShareLib/GpuUtils.cuh"
#include "core/Finisher.h"
#include "satUtils/SolverTypes.h"

namespace Glucose {
    class GpuOptions;
    class GpuHelpedSolver;
}

// Used by tests
namespace GpuShare {

class GpuRunner;
class ContigCopier;

void setDefaultOptions(GpuClauseSharerOptions &options);

class GpuClauseSharerForTests : public GpuClauseSharerImpl {
    public:
    using GpuClauseSharerImpl::assigs;
    using GpuClauseSharerImpl::gpuRunner;
    using GpuClauseSharerImpl::reported;
    using GpuClauseSharerImpl::clauses;
    using GpuClauseSharerImpl::sp;

    GpuClauseSharerForTests(GpuClauseSharerOptions opts);
};

class GpuFixture {
public:
    Glucose::Finisher finisher;
    std::vector<Glucose::GpuHelpedSolver*> solvers;
    GpuClauseSharerForTests gpuClauseSharer;

    GpuFixture(GpuClauseSharerOptions &options, int varCount, int solverCount, int initRepSize = 100);
    ~GpuFixture();

    void execute();
    Glucose::CRef executeAndImportClauses();
    void executeAndImportClauses(std::vector<Glucose::CRef> &res);
    void checkReportedImported(int count, int instance, bool unit);
    void addClause(const std::vector<Lit> &cl);
};

void execute(GpuClauseSharer &gpuClauseSharer);

void copyToDeviceAsync(HostClauses &hCls, cudaStream_t &stream, GpuDims gpuDims);
void copyToDeviceAsync(HostClauses &hCls, cudaStream_t &stream, ContigCopier &cc, GpuDims gpuDims);

void addClause(HostClauses &hostClauses, const std::vector<Lit> &cl);

}

#endif
