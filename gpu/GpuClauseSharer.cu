#include "GpuClauseSharer.cuh"

namespace Glucose {
    GpuClauseSharer(GpuClauseShareOptions opts, /* TODO: we should be able to increase it */ int varCount) {
        GpuDims gpuDims;
        if (opts.gpuBlockCountGuideline > 0) {
            gpuDims.blockCount = opts.gpuBlockCountGuideline;
        } else {
            cudaDeviceProp props;
            exitIfError(cudaGetDeviceProperties(&props, 0), POSITION);
            gpuDims.blockCount = props.multiProcessorCount * 2;
            if (opts.verbosity > 0) printf("c Setting block count guideline to %d (twice the number of multiprocessors)\n", gpuDims.blockCount);
        }
        gpuDims.threadsPerBlock = opts.gpuThreadsPerBlockGuideline;
        assigs = my_make_unique<HostAssigs>(varCount, gpuDims);  
        clauses = my_make_unique<HostClauses>(gpuDims, opts.clauseActivityDecay, true);
        reported = std::make_unique<Reported>(*hClauses);
        gpuRunner = my_make_unique<GpuRunner>(*hClauses, *hostAssigs, *reported, gpuDims, ops.quickProf, initRepCountPerCategory, ops.minGpuLatencyMicros, streamPointer.get());

    }
}
