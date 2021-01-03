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

#include "gpuShareLib/Helper.cuh"
#include "testUtils/TestHelper.cuh"
#include "gpu/CompositionRoot.h"
#include "gpuShareLib/Clauses.cuh"
#include <boost/test/unit_test.hpp>
#include "gpuShareLib/GpuClauseSharer.h"

using namespace Glucose;

namespace GpuShare {

void setDefaultOptions(GpuClauseSharerOptions &options) {
    options.gpuBlockCountGuideline = 3;
    options.gpuThreadsPerBlockGuideline = 32;
    options.minGpuLatencyMicros = 50;
}

GpuFixture::GpuFixture(GpuClauseSharerOptions &options, int varCount, int _solverCount, int initRepCountPerCategory) :
        gpuClauseSharer(options)
{
    gpuClauseSharer.setVarCount(varCount);
    gpuClauseSharer.setCpuSolverCount(_solverCount);
    for (int s = 0; s < _solverCount; s++) {
        GpuHelpedSolver *solv = new GpuHelpedSolver(finisher, s, GpuHelpedSolverParams {true}, gpuClauseSharer, options.quickProf);
        solvers.push(solv);
        for (int i = 0; i < varCount; i++) {
            solv->newVar();
        }
    }
}

GpuClauseSharerForTests::GpuClauseSharerForTests(GpuClauseSharerOptions opts): GpuClauseSharerImpl(opts) {
}

void execute(GpuClauseSharer &gpuClauseSharer) {
    // if we run execute just once, it will start the gpu run but won't 
    // get the results back
    for (int i = 0; i < 2; i++) gpuClauseSharer.gpuRun();
}

void GpuFixture::execute() {
    for (int i = 0; i < solvers.size(); i++) {
        solvers[i] -> tryCopyTrailForGpu(solvers[i] -> decisionLevel());
    }
    GpuShare::execute(gpuClauseSharer);
}

CRef GpuFixture::executeAndImportClauses() {
    assert(solvers.size() == 1);
    vec<CRef> res;
    executeAndImportClauses(res);
    return res[0];
}

void GpuFixture::executeAndImportClauses(vec<CRef> &res) {
    execute();
    bool foundEmptyClause = false;
    res.clear();
    for (int i = 0; i < solvers.size(); i++) {
        res.push(solvers[i]->gpuImportClauses(foundEmptyClause));
    }
}

void GpuFixture::checkReportedImported(int count, int instance, bool unit) {
    BOOST_CHECK_EQUAL(gpuClauseSharer.getOneSolverStat(instance, reportedClauses), count);
    BOOST_CHECK_EQUAL(solvers[instance]->stats[nbImportedValid], count);
    if (unit) {
        BOOST_CHECK_EQUAL(solvers[instance]->stats[nbImportedUnit], count);
    } else {
        BOOST_CHECK_EQUAL(solvers[instance]->stats[nbImported], count);
    }
}

GpuFixture::~GpuFixture() {
    for (int i = 0; i < solvers.size(); i++) {
        delete solvers[i];
    }
}

__global__ void globalUpdateClauses(DClauseUpdates clUpdates, DClauses dClauses) {
    updateClauses(clUpdates, dClauses);
}

// often, this method is called just to make the clause counts on the host clauses right
void copyToDeviceAsync(HostClauses &hCls, cudaStream_t &stream, GpuDims gpuDims) {
    ContigCopier cc;
    copyToDeviceAsync(hCls, stream, cc, gpuDims);
}

void GpuFixture::addClause(const vec<Lit> &cl) {
    gpuClauseSharer.addClause((int*) &cl[0], cl.size());
}

void copyToDeviceAsync(HostClauses &hCls, cudaStream_t &stream, ContigCopier &cc, GpuDims gpuDims) {
    cc.clear(false);
    ClUpdateSet updates = hCls.getUpdatesForDevice(stream, cc);
    RunInfo runInfo = hCls.makeRunInfo(stream, cc);
    exitIfFalse(cc.tryCopyAsync(cudaMemcpyHostToDevice, stream), POSITION);
    // TODO: take GpuDims here
    DClauses dClauses = runInfo.getDClauses();
    globalUpdateClauses<<<gpuDims.blockCount, gpuDims.threadsPerBlock, 0, stream>>>(updates.getDClauseUpdates(), dClauses);
    exitIfError(cudaStreamSynchronize(stream), POSITION);
}

void addClause(HostClauses &hostClauses, const vec<Lit> &cl) {
    hostClauses.addClause(MinHArr<Lit>((size_t) cl.size(), (Lit*) &cl[0]), cl.size());
}

}
