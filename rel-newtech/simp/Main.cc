/*****************************************************************************************[Main.cc]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007,      Niklas Sorensson

Chanseok Oh's MiniSat Patch Series -- Copyright (c) 2015, Chanseok Oh
 
Maple_LCM, Based on MapleCOMSPS_DRUP -- Copyright (c) 2017, Mao Luo, Chu-Min LI, Fan Xiao: implementing a learnt clause minimisation approach
Reference: M. Luo, C.-M. Li, F. Xiao, F. Manya, and Z. L. , “An effective learnt clause minimization approach for cdcl sat solvers,” in IJCAI-2017, 2017, pp. to–appear.

Maple_LCM_Dist, Based on Maple_LCM -- Copyright (c) 2017, Fan Xiao, Chu-Min LI, Mao Luo: using a new branching heuristic called Distance at the beginning of search

MapleLCMDistChronoBT, based on Maple_LCM_Dist -- Copyright (c), Alexander Nadel, Vadim Ryvchin: "Chronological Backtracking" in SAT-2018, pp. 111-121.

MapleLCMDistChronoBT-DL, based on MapleLCMDistChronoBT -- Copyright (c), Stepan Kochemazov, Oleg Zaikin, Victor Kondratiev, Alexander Semenov: The solver was augmented with heuristic that moves duplicate learnt clauses into the core/tier2 tiers depending on a number of parameters.
 
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

#include <errno.h>

#include <signal.h>
#include <zlib.h>
#include <sys/resource.h>
#include <memory>

#include "../utils/System.h"
#include "../utils/ParseUtils.h"
#include "../utils/Options.h"
#include "../utils/JsonWriter.h"
#include "../core/Dimacs.h"
#include "../simp/SimpSolver.h"
#include "InitHelper.h"
#include "gpuShareLib/my_make_unique.h"
#include "gpuShareLib/Utils.h"

using namespace Minisat;

//=================================================================================================


void printStats(Solver& solver)
{
    double cpu_time = cpuTimeSec();
    double mem_used = memUsedPeak();
    printf("c restarts              : %" PRIu64"\n", solver.starts);
    printf("c duplicate learnts_cnf : %" PRIu64"\n", solver.duplicates_added_conflicts);
    printf("c duplicate learnts_min : %" PRIu64"\n", solver.duplicates_added_minimization);
    printf("c conflicts             : %-12" PRIu64"   (%.0f /sec)\n", solver.conflicts   , solver.conflicts   /cpu_time);
    printf("c decisions             : %-12" PRIu64"   (%4.2f %% random) (%.0f /sec)\n", solver.decisions, (float)solver.rnd_decisions*100 / (float)solver.decisions, solver.decisions   /cpu_time);
    printf("c propagations          : %-12" PRIu64"   (%.0f /sec)\n", solver.propagations, solver.propagations/cpu_time);
    printf("c conflict literals     : %-12" PRIu64"   (%4.2f %% deleted)\n", solver.tot_literals, (solver.max_literals - solver.tot_literals)*100 / (double)solver.max_literals);
    printf("c backtracks            : %-12" PRIu64"   (NCB %0.f%% , CB %0.f%%)\n", solver.non_chrono_backtrack + solver.chrono_backtrack, (solver.non_chrono_backtrack * 100) / (double)(solver.non_chrono_backtrack + solver.chrono_backtrack), (solver.chrono_backtrack * 100) / (double)(solver.non_chrono_backtrack + solver.chrono_backtrack));
    if (mem_used != 0) printf("c Memory used           : %.2f MB\n", mem_used);
    printf("c CPU time              : %g s\n", cpu_time);
    printf("c real time             : %g s\n", realTimeSecSinceStart());
}


Finisher finisher;
// Terminate by notifying the solver and back out gracefully. This is mainly to have a test-case
// for this feature of the Solver as it may take longer than an immediate call to '_exit()'.
static void SIGINT_interrupt(int signum) { finisher.stopAllThreads = true; }

static Solver* solver;
// Note that '_exit()' rather than 'exit()' has to be used. The reason is that 'exit()' calls
// destructors and may cause deadlocks if a malloc/free function happens to be running (these
// functions are guarded by locks for multithreaded use).
static void SIGINT_exit(int signum) {
    printf("\n"); printf("c *** INTERRUPTED ***\n");
    if (solver->getVerbosity() > 0){
        printStats(*solver);
        printf("\n"); printf("c *** INTERRUPTED ***\n"); }
    _exit(1); }


//=================================================================================================
// Main:

int main(int argc, char** argv)
{
    try {
        setUsageHelp("USAGE: %s [options] <input-file> <result-output-file>\n\n  where input may be either in plain or gzipped DIMACS.\n");
        printf("c This is MapleLCMDistChronoBT-DL.\n");
        
#if defined(__linux__)
        fpu_control_t oldcw, newcw;
        _FPU_GETCW(oldcw); newcw = (oldcw & ~_FPU_EXTENDED) | _FPU_DOUBLE; _FPU_SETCW(newcw);
        printf("c WARNING: for repeatability, setting FPU to use double precision\n");
#endif
        // Extra options:
        //
        StringOption dimacs ("MAIN", "dimacs", "If given, stop after preprocessing and write the result to this file.");
        BoolOption   drup   ("MAIN", "drup",   "Generate DRUP UNSAT proof.", false);
        StringOption drup_file("MAIN", "drup-file", "DRUP UNSAT proof ouput file.", "");
        CommonOptions commonOptions;

        parseOptions(argc, argv, true);
        GpuShare::GpuClauseSharer gpuClauseSharer; 
        GpuShare::Logger logger {commonOptions.getVerbosity().global, GpuShare::directPrint};
        SimpSolver  S(0, gpuClauseSharer, finisher, commonOptions.quickProf, logger);


        S.parsing = true;
        std::unique_ptr<JsonStatsWriter> statsWriter;
        if (commonOptions.jsonOutPath != NULL) {
            statsWriter = my_make_unique<JsonStatsWriter>(commonOptions.jsonOutPath);
            S.writeStatsPeriodically(20000, statsWriter->writer);
        }
		S.drup_file = NULL;
        if (drup || strlen(drup_file)){
            S.drup_file = strlen(drup_file) ? fopen(drup_file, "wb") : stdout;
            if (S.drup_file == NULL){
                S.drup_file = stdout;
                printf("c Error opening %s for write.\n", (const char*) drup_file); }
            printf("c DRUP proof generation: %s\n", S.drup_file == stdout ? "stdout" : drup_file);
        }

        solver = &S;
        // Use signal handlers that forcibly quit until the solver will be able to respond to
        // interrupts:
        signal(SIGINT, SIGINT_exit);
        signal(SIGXCPU,SIGINT_exit);

        commonOptions.applyTimeAndCpuLim();
        commonOptions.applyMemLim(0);
        if (!commonOptions.doPreprocessing()) S.use_simplification = false;
        
        gzFile in = getInputFile(argc, argv);
        
        if (logger.verb > 0){
            printf("c ============================[ Problem Statistics ]=============================\n");
            printf("c |                                                                             |\n"); }
        {
            TimePrinter tp("Parse", PrintCpu && (logger.verb > 0));
            parse_DIMACS(in, S);
        }
        gzclose(in);
        FILE* res = (argc >= 3) ? fopen(argv[2], "wb") : NULL;

        if (logger.verb > 0){
            printf("c |  Number of variables:  %12d                                         |\n", S.nVars());
            printf("c |  Number of clauses:    %12d                                         |\n", S.nClauses()); }

        if (commonOptions.doPreprocessing()) {
            TimePrinter tp("Simplification", PrintCpu && (logger.verb > 0));
            S.eliminate(true);
        }
        // Change to signal-handlers that will only notify the solver and allow it to terminate
        // voluntarily:
        signal(SIGINT, SIGINT_interrupt);
        signal(SIGXCPU,SIGINT_interrupt);

        S.parsing = false;

        if (!S.okay()){
            if (res != NULL) fprintf(res, "UNSAT\n"), fclose(res);
            if (logger.verb > 0){
                printf("c ===============================================================================\n");
                printf("c Solved by simplification\n");
                printStats(S);
                printf("\n"); }
            printf("s UNSATISFIABLE\n");
            if (S.drup_file){
#ifdef BIN_DRUP
                fputc('a', S.drup_file); fputc(0, S.drup_file);
#else
                fprintf(S.drup_file, "0\n");
#endif
            }
            if (S.drup_file && S.drup_file != stdout) fclose(S.drup_file);
            exit(20);
        }

        if (dimacs){
            if (logger.verb > 0)
                printf("c ==============================[ Writing DIMACS ]===============================\n");
            S.toDimacs((const char*)dimacs);
            if (logger.verb > 0)
                printStats(S);
            exit(0);
        }

        vec<Lit> dummy;
        lbool ret = S.solveLimited(dummy);
        
        if (logger.verb > 0){
            printStats(S);
            printf("\n"); }
        printf(ret == l_True ? "s SATISFIABLE\n" : ret == l_False ? "s UNSATISFIABLE\n" : "s UNKNOWN\n");
        if (ret == l_True){
            // printf("%d",S.nVars());
            printf("v ");
            for (int i = 0; i < S.nVars(); i++)
                if (S.model[i] != l_Undef)
                    printf("%s%s%d", (i==0)?"":" ", (S.model[i]==l_True)?"":"-", i+1);
            printf(" 0\n");
        }


        if (S.drup_file && ret == l_False){
#ifdef BIN_DRUP
            fputc('a', S.drup_file); fputc(0, S.drup_file);
#else
            fprintf(S.drup_file, "0\n");
#endif
        }
        if (S.drup_file && S.drup_file != stdout) fclose(S.drup_file);

        if (res != NULL){
            if (ret == l_True){
                fprintf(res, "SAT\n");
                for (int i = 0; i < S.nVars(); i++)
                    if (S.model[i] != l_Undef)
                        fprintf(res, "%s%s%d", (i==0)?"":" ", (S.model[i]==l_True)?"":"-", i+1);
                fprintf(res, " 0\n");
            }else if (ret == l_False)
                fprintf(res, "UNSAT\n");
            else
                fprintf(res, "INDET\n");
            fclose(res);
        }

#ifdef NDEBUG
        statsWriter.reset();
        exit(ret == l_True ? 10 : ret == l_False ? 20 : 0);     // (faster than "return", which will invoke the destructor for 'Solver')
#else
        return (ret == l_True ? 10 : ret == l_False ? 20 : 0);
#endif
    } catch (OutOfMemoryException&){
        printf("c ===============================================================================\n");
        printf("c Out of memory\n");
        printf("s UNKNOWN\n");
        exit(0);
    }
}
