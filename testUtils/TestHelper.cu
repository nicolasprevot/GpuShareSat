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

#include "gpu/Helper.cuh"
#include "testUtils/TestHelper.cuh"
#include <boost/test/unit_test.hpp>


namespace Glucose {

void setDefaultOptions(GpuOptions &options) {
    options.blockCount = 3;
    options.threadsPerBlock = 32;
    options.gpuActOnly = false;
}

GpuFixture::GpuFixture(GpuOptions options, int varCount, int _solverCount, int initRepCountPerCategory) :
        co(options, finisher, varCount, initRepCountPerCategory),
        solverCount(_solverCount)
{
    int warpsPerBlock = co.gpuDims.threadsPerBlock / WARP_SIZE;
    co.hostAssigs->growSolverAssigs(_solverCount, warpsPerBlock, warpsPerBlock * co.gpuDims.blockCount);
    for (int s = 0; s < _solverCount; s++) {
        GpuHelpedSolver *solv = new GpuHelpedSolver(*co.reported, finisher, *co.hClauses,
            s, options.gpuHelpedSolverOptions.toParams(), co.hostAssigs->getAssigs(s));
        solvers.push(solv);
        for (int i = 0; i < varCount; i++) {
            solv->newVar();
        }
    }
    co.reported->setSolverCount(_solverCount);
}

void execute(GpuRunner &gpuRunner) {
    // if we run execute just once, it will start the gpu run but won't 
    // get the results back
    for (int i = 0; i < 2; i++) gpuRunner.execute();
}

void GpuFixture::execute() {
    for (int i = 0; i < solvers.size(); i++) {
        solvers[i] -> copyAssigsForGpu(solvers[i] -> decisionLevel());
    }
    Glucose::execute(*co.gpuRunner);
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
    BOOST_CHECK_EQUAL(solvers[instance]->stats[nbReported], count);
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

void addClause(HostClauses &hcls, Lit lit, int lbd) {
    vec<Lit> lits;
    lits.push(lit);
    return hcls.addClause(lits, lbd);
}

void addClause(HostClauses &hcls, Lit lit1, Lit lit2, int lbd) {
    vec<Lit> lits;
    lits.push(lit1);
    lits.push(lit2);
    return hcls.addClause(lits, lbd);
}

void addClause(HostClauses &hcls, Lit lit1, Lit lit2, Lit lit3, int lbd) {
    vec<Lit> lits;
    lits.push(lit1);
    lits.push(lit2);
    lits.push(lit3);
    return hcls.addClause(lits, lbd);
}

__global__ void globalUpdateClauses(DClauseUpdates clUpdates, DClauses dClauses) {
    updateClauses(clUpdates, dClauses);
}

// often, this method is called just to make the clause counts on the host clauses right
void copyToDeviceAsync(HostClauses &hCls, cudaStream_t &stream, GpuDims gpuDims) {
    ContigCopier cc;
    copyToDeviceAsync(hCls, stream, cc, gpuDims);
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

}
