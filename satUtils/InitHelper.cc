/***************************************************************************************[Solver.cc]
 Glucose -- Copyright (c) 2009-2014, Gilles Audemard, Laurent Simon
                                CRIL - Univ. Artois, France
                                LRI  - Univ. Paris Sud, France (2009-2013)
                                Labri - Univ. Bordeaux, France

 Syrup (Glucose Parallel) -- Copyright (c) 2013-2014, Gilles Audemard, Laurent Simon
                                CRIL - Univ. Artois, France
                                Labri - Univ. Bordeaux, France

 GpuShareSat -- Copyright (c) 2020, Nicolas Prevot

Glucose sources are based on MiniSat (see below MiniSat copyrights). Permissions and copyrights of
Glucose (sources until 2013, Glucose 3.0, single core) are exactly the same as Minisat on which it 
is based on. (see below).

Glucose-Syrup sources are based on another copyright. Permissions and copyrights for the parallel
version of Glucose-Syrup (the "Software") are granted, free of charge, to deal with the Software
without restriction, including the rights to use, copy, modify, merge, publish, distribute,
sublicence, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

- The above and below copyrights notices and this permission notice shall be included in all
copies or substantial portions of the Software;
- The parallel version of Glucose (all files modified since Glucose 3.0 releases, 2013) cannot
be used in any competitive event (sat competitions/evaluations) without the express permission of 
the authors (Gilles Audemard / Laurent Simon). This is also the case for any competitive event
using Glucose Parallel as an embedded SAT engine (single core or not).


--------------- Original Minisat Copyrights

Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

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

#include <stdio.h>
#include <sys/resource.h>
#include <thread>
#include <chrono>
#include "InitHelper.h"
#include "utils/System.h"
#include <csignal>

using namespace Glucose;

void setCpuLim(int cpu_lim) {
    rlimit rl;
    getrlimit(RLIMIT_CPU, &rl);
    if (rl.rlim_max == RLIM_INFINITY || (rlim_t)cpu_lim < rl.rlim_max) {
        rl.rlim_cur = cpu_lim;
        if (setrlimit(RLIMIT_CPU, &rl) == -1)
            printf("c WARNING! Could not set resource limit: CPU-time.\n");
    }
}

void setMemLim(int mem_lim) {
    rlim_t new_mem_lim = (rlim_t)mem_lim * 1024*1024;
    rlimit rl;
    getrlimit(RLIMIT_AS, &rl);
    if (rl.rlim_max == RLIM_INFINITY || new_mem_lim < rl.rlim_max){
        rl.rlim_cur = new_mem_lim;
        if (setrlimit(RLIMIT_AS, &rl) == -1)
            printf("c WARNING! Could not set resource limit: Virtual memory.\n");
    }
}

CommonOptions::CommonOptions():
    cpuLim("MAIN", "cpu-lim","Limit on CPU time allowed in seconds.\n", INT32_MAX, IntRange(0, INT32_MAX)),
    timeLim("MAIN", "time-lim","Limit on time allowed in seconds.\n", INT32_MAX, IntRange(0, INT32_MAX)),
    memLim("MAIN", "mem-lim","Limit on memory usage in megabytes.\n", 12000, IntRange(0, INT32_MAX)),
    verb("MAIN", "verb",   "Verbosity level (0=silent, 1=some, 2=more).", 1, IntRange(0, 2)),
    mod("MAIN", "model",   "show model.", false),
    pre("MAIN", "pre", "Completely turn on/off any preprocessing.", true),
    vv("MAIN", "vv",   "Verbosity every vv conflicts", 10000, IntRange(1,INT32_MAX))
    {

}

void CommonOptions::applyMemLim(int additionalMemLim) {
    // Set limit on virtual memory:
    if (memLim != INT32_MAX){
        setMemLim(memLim + additionalMemLim);
    }
}

void* killAfterDelay(void *arg) {
    int32_t delaySec;
    memcpy(&delaySec, &arg, sizeof(int32_t));
    std::this_thread::sleep_for(std::chrono::seconds(delaySec));
    // This will trigger the code which is run when the process is interrupted in other ways
    printf("c reached time limit\n");
    raise(SIGINT);
    return NULL;
}

void CommonOptions::applyTimeAndCpuLim() {
    // Set limit on CPU-time:
    if (cpuLim != INT32_MAX){
        setCpuLim(cpuLim);
    }
    if (timeLim != INT32_MAX) {
        pthread_t thread;
        void *ptr;
        int32_t timeLimInt = (int32_t) timeLim;
        memcpy(&ptr, &timeLimInt, sizeof(int32_t));
        pthread_create(&thread, NULL, &killAfterDelay, ptr);
    }
}

Verbosity CommonOptions::getVerbosity() {
    return Verbosity(verb, vv, mod);
}

gzFile Glucose::getInputFile(int argc, char** argv) {
    if (argc == 1)
        printf("c Reading from standard input... Use '--help' for help.\n");

    gzFile in = (argc == 1) ? gzdopen(0, "rb") : gzopen(argv[1], "rb");
    if (in == NULL)
        printf("ERROR! Could not open file: %s\n", argc == 1 ? "<stdin>" : argv[1]), exit(1);
    return in;
}

void Glucose::printResult(lbool ret) {
    printf(ret == l_True ? "s SATISFIABLE\n" : ret == l_False ? "s UNSATISFIABLE\n" : "s INDETERMINATE\n");
}

void Glucose::printModel(FILE* res, const vec<lbool>& model) {
    fprintf(res, "v ");
    for (int i = 0; i < model.size(); i++) {
        assert(model[i] != l_Undef);
        fprintf(res, "%s%s%d", (i==0) ? "" : " ", (model[i] == l_True) ? "" : "-", i + 1);
    }
    fprintf(res, " 0\n");
}

