/***********************************************************************************[SolverTypes.h]
MiniSat -- Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
           Copyright (c) 2007-2010, Niklas Sorensson
 
Chanseok Oh's MiniSat Patch Series -- Copyright (c) 2015, Chanseok Oh

Maple_LCM, Based on MapleCOMSPS_DRUP -- Copyright (c) 2017, Mao Luo, Chu-Min LI, Fan Xiao: implementing a learnt clause minimisation approach
Reference: M. Luo, C.-M. Li, F. Xiao, F. Manya, and Z. L. , “An effective learnt clause minimization approach for cdcl sat solvers,” in IJCAI-2017, 2017, pp. to–appear.
 
Maple_LCM_Dist, Based on Maple_LCM -- Copyright (c) 2017, Fan Xiao, Chu-Min LI, Mao Luo: using a new branching heuristic called Distance at the beginning of search
 
 
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

#ifndef Glucose_InitHelper_h
#define Glucose_InitHelper_h

#include "../utils/Options.h"
#include "../core/SolverTypes.h"

#include <zlib.h>
// This file is used for functions that are used to initialise things

namespace Minisat {

inline int getReturnCode(lbool ret) {
    return (ret == l_True ? 10 : ret == l_False ? 20 : 0);
}

template<typename T> void handleInterrupted(T &solv) {
    if (solv.verbosity() > 0) {
        solv.printEncapsulatedStats();
        printf("\n");
        printf("*** INTERRUPTED ***\n");
    }
    printf("s INDETERMINATE\n");
    exit(1);
}

class CommonOptions {
private:
    IntOption cpuLim;
    IntOption timeLim;
    IntOption memLim;
    IntOption verb;
    BoolOption mod;
    BoolOption pre;
public:
    StringOption jsonOutPath;
    BoolOption quickProf;

    CommonOptions();
    void applyMemLim(int additionalMemLim = 0);
    void applyTimeAndCpuLim();
    Verbosity getVerbosity();
    bool doPreprocessing() {return pre; }
};

gzFile getInputFile(int argc, char** argv);

void printResult(lbool sol);

void printModel(FILE* res, const vec<lbool>& model);

}
#endif
