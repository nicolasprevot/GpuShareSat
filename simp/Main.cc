/***************************************************************************************[Main.cc]
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

#include <errno.h>

#include <signal.h>
#include <sys/resource.h>

#include "core/Finisher.h"
#include "utils/System.h"
#include "utils/ParseUtils.h"
#include "utils/Options.h"
#include "satUtils/Dimacs.h"
#include "simp/SimpSolver.h"
#include "satUtils/InitHelper.h"
#include "gpuShareLib/Utils.h"

using namespace Glucose;

//=================================================================================================

static const char* _certified = "CORE -- CERTIFIED UNSAT";



static SimpSolver* solver;

Finisher finisher;
// Terminate by notifying the solver and back out gracefully. This is mainly to have a test-case
// for this feature of the Solver as it may take longer than an immediate call to '_exit()'.
static void SIGINT_interrupt(int signum) { finisher.stopAllThreads = true; }

static void SIGINT_exit(int signum) {
    handleInterrupted(*solver);
}


//=================================================================================================
// Main:

int main(int argc, char** argv)
{
    try {
      printf("c\nc This is glucose 4.0 --  based on MiniSAT (Many thanks to MiniSAT team)\nc\n");

      TimePrinter timePrinter("taken total");

      setUsageHelp("c USAGE: %s [options] <input-file> <result-output-file>\n\n  where input may be either in plain or gzipped DIMACS.\n");


#if defined(__linux__)
        fpu_control_t oldcw, newcw;
        _FPU_GETCW(oldcw); newcw = (oldcw & ~_FPU_EXTENDED) | _FPU_DOUBLE; _FPU_SETCW(newcw);
        //printf("c WARNING: for repeatability, setting FPU to use double precision\n");
#endif
        // Extra options:
        //
        StringOption dimacs ("MAIN", "dimacs", "If given, stop after preprocessing and write the result to this file.");
 //     BoolOption opt_incremental ("MAIN","incremental", "Use incremental SAT solving",false);

        BoolOption    opt_certified      (_certified, "certified",    "Certified UNSAT using DRUP format", false);
        StringOption  opt_certified_file      (_certified, "certified-output",    "Certified UNSAT output file", "NULL");
        IntOption     opt_verb_every_conflicts("MAIN", "vv",   "Verbosity every vv conflicts", 10000, IntRange(1,INT32_MAX));
        BoolOption    opt_vbyte             (_certified, "vbyte",    "Emit proof in variable-byte encoding", false);
        CommonOptions commonOptions;

        parseOptions(argc, argv, true);

        commonOptions.applyTimeAndCpuLim();
        commonOptions.applyMemLim();
        GpuShare::Logger logger {2, GpuShare::directPrint};
        SimpSolver  S(0, finisher, logger);
        double      initial_time = cpuTimeSec();

        S.parsing = 1;
        S.use_simplification = commonOptions.doPreprocessing();

        //if (!pre) S.eliminate(true);
        Verbosity verb = commonOptions.getVerbosity();
        verb.everyConflicts = opt_verb_every_conflicts;
        S.setVerbosity(verb);

        S.certifiedUNSAT = opt_certified;
        S.vbyte = opt_vbyte;
        if(S.certifiedUNSAT) {
            if(!strcmp(opt_certified_file,"NULL")) {
                S.vbyte =  false;  // Cannot write binary to stdout
                S.certifiedOutput =  fopen("/dev/stdout", "wb");
                if(S.verbosity() >= 1)
                    printf("c\nc Write unsat proof on stdout using text format\nc\n");
            } else {
                S.certifiedOutput =  fopen(opt_certified_file, "wb");
                const char *name = opt_certified_file;
                if(S.verbosity() >= 1)
                    printf("c\nc Write unsat proof on %s using %s format\nc\n",name,S.vbyte ? "binaryClausesary" : "text");
            }
        }

        solver = &S;
        // Use signal handlers that forcibly quit until the solver will be able to respond to
        // interrupts:
        signal(SIGINT, SIGINT_exit);
        signal(SIGXCPU,SIGINT_exit);

        gzFile in = getInputFile(argc, argv);

        printProblemStatsHeader(S);

        FILE* res = (argc >= 3) ? fopen(argv[argc-1], "wb") : NULL;
        DimacsParser dp(in);
        dp.fillSolver(S);
        gzclose(in);

        printVarsClsCount(S);

        double parsed_time = cpuTimeSec();
        printParseTime(S, parsed_time - initial_time);

        // Change to signal-handlers that will only notify the solver and allow it to terminate
        // voluntarily:
        signal(SIGINT, SIGINT_interrupt);
        signal(SIGXCPU,SIGINT_interrupt);

        S.parsing = 0;
        if(commonOptions.doPreprocessing()/* && !S.isIncremental()*/) {
        printf("c | Preprocesing is fully done\n");
        S.eliminate(true);
        double simplified_time = cpuTimeSec();
        printSimpTime(S, simplified_time - parsed_time);

    }
    printf("c |                                                                                                       |\n");
        if (!S.okay()){
            if (S.certifiedUNSAT) fprintf(S.certifiedOutput, "0\n"), fclose(S.certifiedOutput);
            if (res != NULL) fprintf(res, "UNSAT\n"), fclose(res);
            if (S.verbosity() > 0){
             printf("c =========================================================================================================\n");
               printf("Solved by simplification\n");
                S.printEncapsulatedStats();
                printf("\n"); }
            printf("s UNSATISFIABLE\n");
            exit(20);
        }

        if (dimacs){
            if (S.verbosity() > 0)
                printf("c =======================================[ Writing DIMACS ]===============================================\n");
            S.toDimacs((const char*)dimacs);
            if (S.verbosity() > 0)
                S.printEncapsulatedStats();
            exit(0);
        }

        vec<Lit> dummy;
        lbool ret = S.solveLimited(dummy);

        if (S.verbosity() > 0){
            S.printEncapsulatedStats();
            printf("\n"); }
        printResult(ret);

        if (res != NULL){
            if (ret == l_True){
                printf("SAT\n");
                printModel(res, S.model);
            } else {
          if (ret == l_False){
              fprintf(res, "UNSAT\n");
          }
        }
            fclose(res);
        } else {
      if(S.getVerbosity().showModel && ret==l_True) {
          printModel(stdout, S.model);
      }
        }

        return getReturnCode(ret);
    } catch (OutOfMemoryException&){
            printf("c =========================================================================================================\n");
        printf("INDETERMINATE\n");
        exit(0);
    }
}
