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

#include <math.h>
#include <algorithm>
#include <sstream>

#include "utils/System.h"
#include "utils/Utils.h"
#include "mtl/Sort.h"
#include "core/Solver.h"
#include "core/Constants.h"
#include "simp/SimpSolver.h"
#include "utils/JsonWriter.h"
#include "Finisher.h"
#include "gpuShareLib/Utils.h"

using namespace Glucose;


//=================================================================================================
// Statistics
//=================================================================================================



//=================================================================================================
// Options:

static const char *_cat = "CORE";
static const char *_cr = "CORE -- RESTART";
static const char *_cred = "CORE -- REDUCE";
static const char *_cm = "CORE -- MINIMIZE";


static DoubleOption opt_K(_cr, "K", "The constant used to force restart", 0.8, DoubleRange(0, false, 1, false));
static DoubleOption opt_R(_cr, "R", "The constant used to block restart", 1.4, DoubleRange(1, false, 5, false));
static IntOption opt_size_lbd_queue(_cr, "size-lbd-queue", "The size of moving average for LBD (restarts)", 50, IntRange(10, INT32_MAX));
static IntOption opt_size_trail_queue(_cr, "size-trail-queue", "The size of moving average for trail (block restarts)", 5000, IntRange(10, INT32_MAX));

static IntOption opt_first_reduce_db(_cred, "first-reduce-db", "The number of conflicts before the first reduce DB (or the size of leernts if chanseok is used)",
                                     2000, IntRange(0, INT32_MAX));
static IntOption opt_inc_reduce_db(_cred, "inc-reduce-db", "Increment for reduce DB", 300, IntRange(0, INT32_MAX));
static IntOption opt_spec_inc_reduce_db(_cred, "special-inc-reduce-db", "Special increment for reduce DB", 1000, IntRange(0, INT32_MAX));
static IntOption opt_lb_lbd_frozen_clause(_cred, "min-lbd-frozen-clause", "Protect clauses if their LBD decrease and is lower than (for one turn)", 30,
                                          IntRange(0, INT32_MAX));
static BoolOption opt_chanseok_hack(_cred, "chanse-ok",
                                    "Use Chanseok Oh strategy for LBD (keep all LBD<=co and remove half of firstreduceDB other learned clauses", false);
static BoolOption opt_act_only(_cred, "act-only", "Look at activity only when comparing clauses, not lbd.", true);
static BoolOption opt_may_perm_learn(_cred, "may-perm-learn", "If we may permanently learn low lbd clauses after adapting the solver", true);

static IntOption opt_chanseok_limit(_cred, "co", "Chanseok Oh: all learned clauses with LBD<=co are permanent", 5, IntRange(2, INT32_MAX));


static IntOption opt_lb_size_minimzing_clause(_cm, "min-size-minimizing-clause", "The min size required to minimize clause", 30, IntRange(3, INT32_MAX));
static IntOption opt_lb_lbd_minimzing_clause(_cm, "min-lbd-minimizing-clause", "The min LBD required to minimize clause", 6, IntRange(3, INT32_MAX));


static DoubleOption opt_var_decay(_cat, "var-decay", "The variable activity decay factor (starting point)", 0.8, DoubleRange(0, false, 1, false));
static DoubleOption opt_max_var_decay(_cat, "max-var-decay", "The variable activity decay factor", 0.95, DoubleRange(0, false, 1, false));
static DoubleOption opt_clause_decay(_cat, "cla-decay", "The clause activity decay factor", 0.999, DoubleRange(0, false, 1, false));
static DoubleOption opt_random_var_freq(_cat, "rnd-freq", "The frequency with which the decision heuristic tries to choose a random variable", 0,
                                        DoubleRange(0, true, 1, true));
static DoubleOption opt_random_seed(_cat, "rnd-seed", "Used by the random variable selection", 91648253, DoubleRange(0, false, HUGE_VAL, false));
static IntOption opt_ccmin_mode(_cat, "ccmin-mode", "Controls conflict clause minimization (0=none, 1=basic, 2=deep)", 2, IntRange(0, 2));
static IntOption opt_phase_saving(_cat, "phase-saving", "Controls the level of phase saving (0=none, 1=limited, 2=full)", 2, IntRange(0, 2));
static BoolOption opt_rnd_init_act(_cat, "rnd-init", "Randomize the initial activity", false);
static DoubleOption opt_garbage_frac(_cat, "gc-frac", "The fraction of wasted memory allowed before a garbage collection is triggered", 0.20,
                                     DoubleRange(0, false, HUGE_VAL, false));
static BoolOption opt_glu_reduction(_cat, "gr", "glucose strategy to fire clause database reduction (must be false to fire Chanseok strategy)", true);
static BoolOption opt_luby_restart(_cat, "luby", "Use the Luby restart sequence", false);
static DoubleOption opt_restart_inc(_cat, "rinc", "Restart interval increase factor", 2, DoubleRange(1, false, HUGE_VAL, false));
static IntOption opt_luby_restart_factor(_cred, "luby-factor", "Luby restart factor", 100, IntRange(1, INT32_MAX));

static IntOption opt_randomize_phase_on_restarts(_cat, "phase-restart",
                                                 "The amount of randomization for the phase at each restart (0=none, 1=first branch, 2=first branch (no bad clauses), 3=first branch (only initial clauses)",
                                                 0, IntRange(0, 3));
static BoolOption opt_fixed_randomize_phase_on_restarts(_cat, "fix-phas-rest", "Fixes the first 7 levels at random phase", false);

static BoolOption opt_adapt(_cat, "adapt", "Adapt dynamically stategies after 100000 conflicts", true);

static BoolOption opt_forceunsat(_cat,"force-unsat","Force the phase for UNSAT",true);
//=================================================================================================
// Constructor/Destructor:

Solver::Solver(int _cpuThreadId, Finisher &_finisher, const GpuShare::Logger &_logger) :

// Parameters (user settable):
//
  K(opt_K)
, R(opt_R)
, sizeLBDQueue(opt_size_lbd_queue)
, sizeTrailQueue(opt_size_trail_queue)
, firstReduceDB(opt_first_reduce_db)
, incReduceDB(opt_chanseok_hack ? 0 : opt_inc_reduce_db)
, specialIncReduceDB(opt_chanseok_hack ? 0 : opt_spec_inc_reduce_db)
, lbLBDFrozenClause(opt_lb_lbd_frozen_clause)
, mayPermLearnLowLbd(opt_may_perm_learn)
, compareLbd(!opt_chanseok_hack && !opt_act_only)
, coLBDBound (opt_chanseok_hack ? opt_chanseok_limit : 2)
, lbSizeMinimizingClause(opt_lb_size_minimzing_clause)
, lbLBDMinimizingClause(opt_lb_lbd_minimzing_clause)
, var_decay(opt_var_decay)
, max_var_decay(opt_max_var_decay)
, clause_decay(opt_clause_decay)
, random_var_freq(opt_random_var_freq)
, random_seed(opt_random_seed)
, ccmin_mode(opt_ccmin_mode)
, phase_saving(opt_phase_saving)
, rnd_pol(false)
, rnd_init_act(opt_rnd_init_act)
, randomizeFirstDescent(false)
, garbage_frac(opt_garbage_frac)
, certifiedOutput(NULL)
, certifiedUNSAT(false) // Not in the first parallel version
, vbyte(false)
, panicModeLastRemoved(0), panicModeLastRemovedShared(0)
, conflictsRestarts(0)
, conflicts(0)
, finisher(_finisher)
, curRestart(1)
, glureduce(opt_glu_reduction)
, restart_inc(opt_restart_inc)
, luby_restart(opt_luby_restart)
, adaptStrategies(opt_adapt)
, luby_restart_factor(opt_luby_restart_factor)
, randomize_on_restarts(opt_randomize_phase_on_restarts)
, fixed_randomize_on_restarts(opt_fixed_randomize_phase_on_restarts)
, newDescent(0)
, randomDescentAssignments(0)
, forceUnsatOnNewDescent(opt_forceunsat)
, ok(true)
, cla_inc(1)
, var_inc(1)
, watches(WatcherDeleted(ca))
, watchesBin(WatcherDeleted(ca))
, qhead(0)
, simpDB_assigns(-1)
, simpDB_props(0)
, order_heap(VarOrderLt(activity))
, progress_estimate(0)
, remove_satisfied(true)
// Resource constraints:
//
, conflict_budget(-1)
, propagation_budget(-1)
, incremental(false)
, nbVarsInitialFormula(INT32_MAX)
, totalTime4Sat(0.)
, totalTime4Unsat(0.)
, nbSatCalls(0)
, nbUnsatCalls(0)
, cpuThreadId(_cpuThreadId)
, learnedPermLearnedImplying(0)
, logger(_logger)
{
    MYFLAG = 0;
    // Initialize only first time. Useful for incremental solving (not in // version), useless otherwise
    // Kept here for simplicity
    lbdQueue.initSize(sizeLBDQueue);
    trailQueue.initSize(sizeTrailQueue);
    sumLBD = 0;
    nbclausesbeforereduce = firstReduceDB;
#define X(v) stats.push(0);
#include "CoreSolverStats.h"
#undef X
	insertStatNames();
}

//-------------------------------------------------------
// Special constructor used for cloning solvers
//-------------------------------------------------------

Solver::Solver(const Solver &s, int _cpuThreadId) :

  K(s.K)
, R(s.R)
, sizeLBDQueue(s.sizeLBDQueue)
, sizeTrailQueue(s.sizeTrailQueue)
, firstReduceDB(s.firstReduceDB)
, incReduceDB(s.incReduceDB)
, specialIncReduceDB(s.specialIncReduceDB)
, lbLBDFrozenClause(s.lbLBDFrozenClause)
, mayPermLearnLowLbd(s.mayPermLearnLowLbd)
, compareLbd(s.compareLbd)
, coLBDBound (s.coLBDBound)
, lbSizeMinimizingClause(s.lbSizeMinimizingClause)
, lbLBDMinimizingClause(s.lbLBDMinimizingClause)
, var_decay(s.var_decay)
, max_var_decay(s.max_var_decay)
, clause_decay(s.clause_decay)
, random_var_freq(s.random_var_freq)
, random_seed(s.random_seed)
, ccmin_mode(s.ccmin_mode)
, phase_saving(s.phase_saving)
, rnd_pol(s.rnd_pol)
, rnd_init_act(s.rnd_init_act)
, randomizeFirstDescent(s.randomizeFirstDescent)
, garbage_frac(s.garbage_frac)
, certifiedOutput(NULL)
, certifiedUNSAT(false) // Not in the first parallel version
, panicModeLastRemoved(s.panicModeLastRemoved), panicModeLastRemovedShared(s.panicModeLastRemovedShared)
// Statistics: (formerly in 'SolverStats')
//
, conflictsRestarts(0)
, conflicts(s.conflicts)
, finisher(s.finisher)
, verb(s.verb)
, curRestart(s.curRestart)
, glureduce(s.glureduce)
, restart_inc(s.restart_inc)
, luby_restart(s.luby_restart)
, adaptStrategies(s.adaptStrategies)
, luby_restart_factor(s.luby_restart_factor)
, randomize_on_restarts(s.randomize_on_restarts)
, fixed_randomize_on_restarts(s.fixed_randomize_on_restarts)
, newDescent(s.newDescent)
, randomDescentAssignments(s.randomDescentAssignments)
, forceUnsatOnNewDescent(s.forceUnsatOnNewDescent)
, ok(true)
, cla_inc(s.cla_inc)
, var_inc(s.var_inc)
, watches(WatcherDeleted(ca))
, watchesBin(WatcherDeleted(ca))
, qhead(s.qhead)
, simpDB_assigns(s.simpDB_assigns)
, simpDB_props(s.simpDB_props)
, order_heap(VarOrderLt(activity))
, progress_estimate(s.progress_estimate)
, remove_satisfied(s.remove_satisfied)
// Resource constraints:
//
, conflict_budget(s.conflict_budget)
, propagation_budget(s.propagation_budget)
, incremental(s.incremental)
, nbVarsInitialFormula(s.nbVarsInitialFormula)
, totalTime4Sat(s.totalTime4Sat)
, totalTime4Unsat(s.totalTime4Unsat)
, nbSatCalls(s.nbSatCalls)
, nbUnsatCalls(s.nbUnsatCalls)
, cpuThreadId(_cpuThreadId)
, learnedPermLearnedImplying(s.learnedPermLearnedImplying)
, logger(s.logger)
{
    // Copy clauses.
    s.ca.copyTo(ca);
    ca.extra_clause_field = s.ca.extra_clause_field;

    // Initialize  other variables
    MYFLAG = 0;
    // Initialize only first time. Useful for incremental solving (not in // version), useless otherwise
    // Kept here for simplicity
    sumLBD = s.sumLBD;
    nbclausesbeforereduce = s.nbclausesbeforereduce;

    // Copy all search vectors
    s.watches.copyTo(watches);
    s.watchesBin.copyTo(watchesBin);
    s.assigns.memCopyTo(assigns);
    s.vardata.memCopyTo(vardata);
    s.activity.memCopyTo(activity);
    s.seen.memCopyTo(seen);
    s.permDiff.memCopyTo(permDiff);
    s.polarity.memCopyTo(polarity);
    s.decision.memCopyTo(decision);
    s.trail.memCopyTo(trail);
    s.order_heap.copyTo(order_heap);
    s.clauses.memCopyTo(clauses);
    s.learned.memCopyTo(learned);
    s.permanentlyLearned.memCopyTo(permanentlyLearned);

    s.lbdQueue.copyTo(lbdQueue);
    s.trailQueue.copyTo(trailQueue);
    s.forceUNSAT.copyTo(forceUNSAT);
    s.stats.copyTo(stats);

    statNames = s.statNames;
}

void Solver::insertStatNames() {
#define X(v) statNames[v] = std::string(#v);
#include "CoreSolverStats.h"
#undef X
}

Solver::~Solver() {
}


/****************************************************************
 Certified UNSAT proof in binary format
****************************************************************/


void Solver::write_char(unsigned char ch) {
    if(putc_unlocked((int) ch, certifiedOutput) == EOF)
        exit(1);
}


void Solver::write_lit(int n) {
    for(; n > 127; n >>= 7)
        write_char(128 | (n & 127));
    write_char(n);
}

/****************************************************************
 Set the incremental mode
****************************************************************/

// This function set the incremental mode to true.
// You can add special code for this mode here.

void Solver::setIncrementalMode() {
#ifdef INCREMENTAL
    incremental = true;
#else
    fprintf(stderr, "c Trying to set incremental mode, but not compiled properly for this.\n");
    exit(1);
#endif
}


// Number of variables without selectors
void Solver::initNbInitialVars(int nb) {
    nbVarsInitialFormula = nb;
}


bool Solver::isIncremental() {
    return incremental;
}


//=================================================================================================
// Minor methods:


// Creates a new SAT variable in the solver. If 'decision' is cleared, variable will not be
// used as a decision variable (NOTE! This has effects on the meaning of a SATISFIABLE result).
//

Var Solver::newVar(bool sign, bool dvar) {
    int v = nVars();
    watches.init(mkLit(v, false));
    watches.init(mkLit(v, true));
    watchesBin.init(mkLit(v, false));
    watchesBin.init(mkLit(v, true));
    assigns.push(l_Undef);
    vardata.push(mkVarData(CRef_Undef, 0));
    activity.push(rnd_init_act ? drand(random_seed) * 0.00001 : 0);
    seen.push(0);
    permDiff.push(0);
    polarity.push(sign);
    forceUNSAT.push(0);
    decision.push();
    trail.capacity(v + 1);
    setDecisionVar(v, dvar);
    return v;
}


bool Solver::addClause_(vec <Lit> &ps) {
    assert(decisionLevel() == 0);
    if(!ok) return false;

    // Check if clause is satisfied and remove false/duplicate literals:
    sort(ps);

    vec <Lit> oc;
    oc.clear();

    Lit p;
    int i, j, flag = 0;
    if(certifiedUNSAT) {
        for(i = j = 0, p = lit_Undef; i < ps.size(); i++) {
            oc.push(ps[i]);
            if(value(ps[i]) == l_True || ps[i] == ~p || value(ps[i]) == l_False)
                flag = 1;
        }
    }

    for(i = j = 0, p = lit_Undef; i < ps.size(); i++)
        if(value(ps[i]) == l_True || ps[i] == ~p)
            // the clause is already satisfied
            return true;
        else if(value(ps[i]) != l_False && ps[i] != p)
            ps[j++] = p = ps[i];
    ps.shrink(i - j);

    if(flag && (certifiedUNSAT)) {
        if(vbyte) {
            write_char('a');
            for(i = j = 0, p = lit_Undef; i < ps.size(); i++)
                write_lit(2 * (var(ps[i]) + 1) + sign(ps[i]));
            write_lit(0);

            write_char('d');
            for(i = j = 0, p = lit_Undef; i < oc.size(); i++)
                write_lit(2 * (var(oc[i]) + 1) + sign(oc[i]));
            write_lit(0);
        }
        else {
            for(i = j = 0, p = lit_Undef; i < ps.size(); i++)
                fprintf(certifiedOutput, "%i ", (var(ps[i]) + 1) * (-2 * sign(ps[i]) + 1));
            fprintf(certifiedOutput, "0\n");

            fprintf(certifiedOutput, "d ");
            for(i = j = 0, p = lit_Undef; i < oc.size(); i++)
                fprintf(certifiedOutput, "%i ", (var(oc[i]) + 1) * (-2 * sign(oc[i]) + 1));
            fprintf(certifiedOutput, "0\n");
        }
    }


    if(ps.size() == 0)
        return ok = false;
    else if(ps.size() == 1) {
        uncheckedEnqueue(ps[0]);
        return ok = (propagate() == CRef_Undef);
    } else {
        CRef cr = ca.alloc(ps, false, false, false);
        clauses.push(cr);
        attachClause(cr);
    }

    return true;
}


void Solver::attachClause(CRef cr) {
    const Clause &c = ca[cr];

    assert(c.size() > 1);
#ifdef DEBUG
    int minOfTwo = std::min(litLevel(c[0]), litLevel(c[1]));
    if (minOfTwo > 0) {
        for (int i = 2; i < c.size(); i++) {
            if (litLevel(c[i]) > minOfTwo) {
                printf("min of two: %d\n", minOfTwo);
                printf("lit level: %d\n", litLevel(c[i]));
            }
            assert(litLevel(c[i]) <= minOfTwo);
        }
    }
#endif
    if(c.size() == 2) {
        watchesBin[~c[0]].push(Watcher(cr, c[1]));
        watchesBin[~c[1]].push(Watcher(cr, c[0]));
    } else {
        watches[~c[0]].push(Watcher(cr, c[1]));
        watches[~c[1]].push(Watcher(cr, c[0]));
    }
    if(c.learned()) stats[learnedLiterals] += c.size();
    else stats[clauseLiterals] += c.size();
}

void Solver::detachClause(CRef cr, bool strict) {
    const Clause &c = ca[cr];

    assert(c.size() > 1);
    if(c.size() == 2) {
        if(strict) {
            remove(watchesBin[~c[0]], Watcher(cr, c[1]));
            remove(watchesBin[~c[1]], Watcher(cr, c[0]));
        } else {
            // Lazy detaching: (NOTE! Must clean all watcher lists before garbage collecting this clause)
            watchesBin.smudge(~c[0]);
            watchesBin.smudge(~c[1]);
        }
    } else {
        if(strict) {
            remove(watches[~c[0]], Watcher(cr, c[1]));
            remove(watches[~c[1]], Watcher(cr, c[0]));
        } else {
            // Lazy detaching: (NOTE! Must clean all watcher lists before garbage collecting this clause)
            watches.smudge(~c[0]);
            watches.smudge(~c[1]);
        }
    }
    if(c.learned()) stats[learnedLiterals] -= c.size();
    else stats[clauseLiterals] -= c.size();
}

void Solver::updateStatsForClauseChanged(Clause &cl, int diff) {
     if (cl.learned()) {
        if (cl.fromGpu()) {
            stats[learnedFromGpu] += diff;
        }
        else {
            stats[learnedNotFromGpu] += diff;
        }
    } else if (cl.permLearned()) {
        if (cl.fromGpu()) {
            stats[permanentlyLearnedFromGpu] += diff;
        }
        else {
            stats[permanentlyLearnedNotFromGpu] += diff;
        }
    }
}

void Solver::removeClause(CRef cr) {
    Clause &c = ca[cr];
#ifdef PRINT_DETAILS_CLAUSES
    {
        if (c.learned()) {
            SyncOut so;
            std::cout << "delete: thread" << cpuThreadId << "" << c << std::endl;
        }
    }
#endif
    updateStatsForClauseChanged(c, -1);
    stats[removedClauses]++;
    if(certifiedUNSAT) {
        if(vbyte) {
            write_char('d');
            for(int i = 0; i < c.size(); i++)
                write_lit(2 * (var(c[i]) + 1) + sign(c[i]));
            write_lit(0);
        }
        else {
            fprintf(certifiedOutput, "d ");
            for(int i = 0; i < c.size(); i++)
                fprintf(certifiedOutput, "%i ", (var(c[i]) + 1) * (-2 * sign(c[i]) + 1));
            fprintf(certifiedOutput, "0\n");
        }
    }

    detachClause(cr);
    // Don't leave pointers to free'd memory!
    if(locked(c)) vardata[var(c[0])].reason = CRef_Undef;
    c.mark(1);
    ca.free(cr);
}


bool Solver::satisfied(const Clause &c) const {
#ifdef INCREMENTAL
    if(incremental)
        return (value(c[0]) == l_True) || (value(c[1]) == l_True);
#endif

    // Default mode
    for(int i = 0; i < c.size(); i++)
        if(value(c[i]) == l_True)
            return true;
    return false;
}



/******************************************************************
 * Minimisation with binary reolution
 ******************************************************************/
void Solver::minimisationWithBinaryResolution(vec <Lit> &out_learned) {

    // Find the LBD measure
    unsigned int lbd = computeLBD(out_learned);
    Lit p = ~out_learned[0];

    if(lbd <= lbLBDMinimizingClause) {
        MYFLAG++;

        for(int i = 1; i < out_learned.size(); i++) {
            permDiff[var(out_learned[i])] = MYFLAG;
        }

        vec <Watcher> &wbin = watchesBin[p];
        int nb = 0;
        for(int k = 0; k < wbin.size(); k++) {
            Lit imp = wbin[k].blocker;
            if(permDiff[var(imp)] == MYFLAG && value(imp) == l_True) {
                nb++;
                permDiff[var(imp)] = MYFLAG - 1;
            }
        }
        int l = out_learned.size() - 1;
        if(nb > 0) {
            stats[reducedClauses]++;
            for(int i = 1; i < out_learned.size() - nb; i++) {
                if(permDiff[var(out_learned[i])] != MYFLAG) {
                    Lit p = out_learned[l];
                    out_learned[l] = out_learned[i];
                    out_learned[i] = p;
                    l--;
                    i--;
                }
            }

            out_learned.shrink(nb);

        }
    }
}

// Revert to the state at given level (keeping all assignment at 'level' but not beyond).
//

void Solver::cancelUntil(int level) {
    if(decisionLevel() > level) {
        for(int c = trail.size() - 1; c >= trail_lim[level]; c--) {
            Var x = var(trail[c]);
#ifdef KEEP_IMPL_COUNT
            CRef cref = reason(x);
            if (cref != CRef_Undef) {
                Clause &cl = ca[cref];
                if (cl.learned() || cl.permLearned()) {
                    learnedPermLearnedImplying--;
                }
            }
#endif
            assigns[x] = l_Undef;
            if(phase_saving > 1 || ((phase_saving == 1) && c > trail_lim.last())) {
                polarity[x] = sign(trail[c]);
            }
            insertVarOrder(x);
        }
        qhead = trail_lim[level];
        trail.shrink(trail.size() - trail_lim[level]);
        trail_lim.shrink(trail_lim.size() - level);
#ifdef PRINT_TRAIL_LENGTH
        if (decisionLevel() > 1) {
            printf("cancel_until_trail: %d\n", trail.size());
        }
#endif
    }
}


//=================================================================================================
// Major methods:

Lit Solver::pickBranchLit() {
    Var next = var_Undef;

    // Random decision:
    if(((randomizeFirstDescent && conflicts == 0) || drand(random_seed) < random_var_freq) && !order_heap.empty()) {
        next = order_heap[irand(random_seed, order_heap.size())];
        if(value(next) == l_Undef && decision[next])
            stats[rndDecisions]++;
    }

    // Activity based decision:
    while(next == var_Undef || value(next) != l_Undef || !decision[next])
        if(order_heap.empty()) {
            next = var_Undef;
            break;
        } else {
            next = order_heap.removeMin();
        }

    if(randomize_on_restarts && !fixed_randomize_on_restarts && newDescent && (decisionLevel() % 2 == 0)) {
        return mkLit(next, (randomDescentAssignments >> (decisionLevel() % 32)) & 1);
    }

    if(fixed_randomize_on_restarts && decisionLevel() < 7) {
        return mkLit(next, (randomDescentAssignments >> (decisionLevel() % 32)) & 1);
    }

    if(next == var_Undef) return lit_Undef;

    if(forceUnsatOnNewDescent && newDescent) {
        if(forceUNSAT[next] != 0)
            return mkLit(next, forceUNSAT[next] < 0);
        return mkLit(next, polarity[next]);

    }

    return next == var_Undef ? lit_Undef : mkLit(next, rnd_pol ? drand(random_seed) < 0.5 : polarity[next]);
}

/*_________________________________________________________________________________________________
|
|  analyze : (confl : Clause*) (out_learned : vec<Lit>&) (out_btlevel : int&)  ->  [void]
|  
|  Description:
|    Analyze conflict and produce a reason clause.
|  
|    Pre-conditions:
|      * 'out_learned' is assumed to be cleared.
|      * Current decision level must be greater than root level.
|  
|    Post-conditions:
|      * 'out_learned[0]' is the asserting literal at level 'out_btlevel'.
|      * If out_learned.size() > 1 then 'out_learned[1]' has the greatest decision level of the 
|        rest of literals. There may be others from the same level though.
|  
|________________________________________________________________________________________________@*/
void Solver::analyze(CRef confl, vec <Lit> &out_learned, vec <Lit> &selectors, int &out_btlevel,
        unsigned int &lbd, unsigned int &szWithoutSelectors) {
    int pathC = 0;
    Lit p = lit_Undef;
    stats[sumConflictLevel] += decisionLevel();
    stats[sumConflictTrail] += trail.size();
#ifdef KEEP_IMPL_COUNT
    stats[sumConflictImplying] += learnedPermLearnedImplying;
#endif
    // Note: if a variable is implied by a clause: it is at the position 0 in this clause
    // Generate conflict clause:
    //
    out_learned.push(); // (leave room for the asserting literal)
    int index = trail.size() - 1;
    int origIndex = index;
    do {
        assert(confl != CRef_Undef); // (otherwise should be UIP)
        Clause &c = ca[confl];
        // Special case for binary clauses
        // The first one has to be SAT
        if(p != lit_Undef && c.size() == 2 && value(c[0]) == l_False) {
            assert(value(c[1]) == l_True);
            Lit tmp = c[0];
            c[0] = c[1], c[1] = tmp;
        }

        if (c.learned()) {
            claBumpActivity(c);
            if (c.fromGpu()) {
                stats[learnedFromGpuSeen]++;
            } else {
                stats[learnedNotFromGpuSeen]++;
            }
#ifdef PRINT_DETAILS_CLAUSES
            {
                SyncOut so;
                std::cout << "used: thread " << cpuThreadId << " " << c << std::endl;
            }
#endif
        } else if (c.permLearned()) {
            if (c.fromGpu()) {
                stats[permanentlyLearnedFromGpuSeen]++;
            } else {
                stats[permanentlyLearnedNotFromGpuSeen]++;
            }
        } else { // original clause
            stats[originalSeen]++;
            c.setSeen(true);
        }
        // DYNAMIC NBLEVEL trick (see competition'09 companion paper)
        if(c.learned() && c.lbd() > 2) {
            unsigned int nblevels = computeLBD(c);
            if(nblevels + 1 < c.lbd()) { // improve the LBD
                if(c.lbd() <= lbLBDFrozenClause) {
                    // seems to be interesting : keep it for the next round
                    c.setCanBeDel(false);
                }
                if ((int) nblevels <= coLBDBound) {
                    // At this point, the clause may be unary watched or not
                    learned.remove(confl);
                    permanentlyLearned.push(confl);
                    updateStatsForClauseChanged(c, -1);
                    c.setPermLearned();
                    updateStatsForClauseChanged(c, 1);
                } else {
#ifdef PRINT_DETAILS_CLAUSES
                    SyncOut so;
                    std::cout << "lbd_change: thread " << cpuThreadId << " " << nblevels << " " << c << std::endl;
#endif
                    c.setLBD(nblevels); // Update it
                }
            }
        }

        for(int j = (p == lit_Undef) ? 0 : 1; j < c.size(); j++) {
            Lit q = c[j];

            if(!seen[var(q)]) {
                if(level(var(q)) == 0) {
                } else { // Here, the old case
                    if(!isSelector(var(q)))
                        varBumpActivity(var(q));

                    // This variable was responsible for a conflict,
                    // consider it as a UNSAT assignation for this literal
                    bumpForceUNSAT(~q); // Negation because q is false here

                    seen[var(q)] = 1;
                    if (level(var(q)) > decisionLevel()) {
                        printf("variable level: %d should be smaller than decision level: %d\n",
                                level(var(q)), decisionLevel());
                    }
                    assert(level(var(q)) <= decisionLevel());
                    if(level(var(q)) >= decisionLevel()) {
                        pathC++;
                        // UPDATEVARACTIVITY trick (see competition'09 companion paper)
                        if(!isSelector(var(q)) && (reason(var(q)) != CRef_Undef) && ca[reason(var(q))].learned())
                            lastDecisionLevel.push(q);
                    } else {
                        if(isSelector(var(q))) {
                            assert(value(q) == l_False);
                            selectors.push(q);
                        } else
                            out_learned.push(q);
                    }
                }
            } //else stats[sumResSeen]++;
        }
        // Select next clause to look at:
        while (!seen[var(trail[index--])]);
        p = trail[index + 1];
        if (level(var(p)) < decisionLevel()) {
            printf("last one we looked at: %d\n", index + 1);
            printf("orig index: %d\n", origIndex);
            printf("level is %d\n", level(var(p)));
            printf("patc is %d\n", pathC);
            printf("decision level is %d\n", decisionLevel());
            for (int i = 0; i < trail.size(); i++) {
                Var v = var(trail[i]);
                printf("var %d in trail: level %d, seen %d\n", i, level(v), seen[v]);
                if (i < trail.size() - 1) assert(level(var(trail[i])) <= level(var(trail[i + 1])));
            }
        }
        assert(level(var(p)) >= decisionLevel());
        //stats[sumRes]++;
        confl = reason(var(p));
        seen[var(p)] = 0;
        pathC--;

    } while(pathC > 0);
    out_learned[0] = ~p;

    // Simplify conflict clause:
    //
    int i, j;

    for(int i = 0; i < selectors.size(); i++)
        out_learned.push(selectors[i]);

    stats[totLearnedLiteralsBeforeMinimize] += out_learned.size();
    out_learned.copyTo(analyze_toclear);
    if(ccmin_mode == 2) {
        uint32_t abstract_level = 0;
        for(i = 1; i < out_learned.size(); i++)
            abstract_level |= abstractLevel(var(out_learned[i])); // (maintain an abstraction of levels involved in conflict)

        for(i = j = 1; i < out_learned.size(); i++)
            if(reason(var(out_learned[i])) == CRef_Undef || !litRedundant(out_learned[i], abstract_level))
                out_learned[j++] = out_learned[i];

    } else if(ccmin_mode == 1) {
        for(i = j = 1; i < out_learned.size(); i++) {
            Var x = var(out_learned[i]);

            if(reason(x) == CRef_Undef)
                out_learned[j++] = out_learned[i];
            else {
                Clause &c = ca[reason(var(out_learned[i]))];
                // Thanks to Siert Wieringa for this bug fix!
                for(int k = ((c.size() == 2) ? 0 : 1); k < c.size(); k++)
                    if(!seen[var(c[k])] && level(var(c[k])) > 0) {
                        out_learned[j++] = out_learned[i];
                        break;
                    }
            }
        }
    } else
        i = j = out_learned.size();

    //    stats[maxLiterals]+=out_learned.size();
    out_learned.shrink(i - j);


    /* ***************************************
      Minimisation with binary clauses of the asserting clause
      First of all : we look for small clauses
      Then, we reduce clauses with small LBD.
      Otherwise, this can be useless
     */
    if(!incremental && out_learned.size() <= lbSizeMinimizingClause) {
        minimisationWithBinaryResolution(out_learned);
    }
    stats[totLearnedLiterals] += out_learned.size();
    // Find correct backtrack level:
    //
    if(out_learned.size() == 1)
        out_btlevel = 0;
    else {
        int max_i = 1;
        // Find the first literal assigned at the next-highest level:
        for(int i = 2; i < out_learned.size(); i++)
            if(level(var(out_learned[i])) > level(var(out_learned[max_i])))
                max_i = i;
        // Swap-in this literal at index 1:
        Lit p = out_learned[max_i];
        out_learned[max_i] = out_learned[1];
        out_learned[1] = p;
        out_btlevel = level(var(p));
    }
#ifdef INCREMENTAL
    if(incremental) {
       szWithoutSelectors = 0;
       for(int i=0;i<out_learned.size();i++) {
     if(!isSelector(var((out_learned[i])))) szWithoutSelectors++;
     else if(i>0) break;
       }
     } else
#endif
    szWithoutSelectors = out_learned.size();
    // Compute LBD
    lbd = computeLBD(out_learned, out_learned.size() - selectors.size());

    // UPDATEVARACTIVITY trick (see competition'09 companion paper)
    if(lastDecisionLevel.size() > 0) {
        for(int i = 0; i < lastDecisionLevel.size(); i++) {
            if(ca[reason(var(lastDecisionLevel[i]))].lbd() < lbd)
                varBumpActivity(var(lastDecisionLevel[i]));
        }
        lastDecisionLevel.clear();
    }


    for(int j = 0; j < analyze_toclear.size(); j++) seen[var(analyze_toclear[j])] = 0; // ('seen[]' is now cleared)
    for(int j = 0; j < selectors.size(); j++) seen[var(selectors[j])] = 0;
}


// Check if 'p' can be removed. 'abstract_levels' is used to abort early if the algorithm is
// visiting literals at levels that cannot be removed later.
// This algorithm works by checking if what recursively implies this literal is part
// of the clause
// in this method, seen is true if a variable is recursively implied by the clause
bool Solver::litRedundant(Lit p, uint32_t abstract_levels) {
    analyze_stack.clear();
    analyze_stack.push(p);
    int top = analyze_toclear.size();
    while(analyze_stack.size() > 0) {
        assert(reason(var(analyze_stack.last())) != CRef_Undef);
        Clause &c = ca[reason(var(analyze_stack.last()))];
        analyze_stack.pop(); //
        if(c.size() == 2 && value(c[0]) == l_False) {
            assert(value(c[1]) == l_True);
            Lit tmp = c[0];
            c[0] = c[1], c[1] = tmp;
        }

        for(int i = 1; i < c.size(); i++) {
            Lit p = c[i];
            if(!seen[var(p)]) {
                if(level(var(p)) > 0) {
                    if(reason(var(p)) != CRef_Undef && (abstractLevel(var(p)) & abstract_levels) != 0) {
                        seen[var(p)] = 1;
                        analyze_stack.push(p);
                        analyze_toclear.push(p);
                    } else {
                        for(int j = top; j < analyze_toclear.size(); j++)
                            seen[var(analyze_toclear[j])] = 0;
                        analyze_toclear.shrink(analyze_toclear.size() - top);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}


/*_________________________________________________________________________________________________
|
|  analyzeFinal : (p : Lit)  ->  [void]
|  
|  Description:
|    Specialized analysis procedure to express the final conflict in terms of assumptions.
|    Calculates the (possibly empty) set of assumptions that led to the assignment of 'p', and
|    stores the result in 'out_conflict'.
|________________________________________________________________________________________________@*/
void Solver::analyzeFinal(Lit p, vec <Lit> &out_conflict) {
    out_conflict.clear();
    out_conflict.push(p);

    if(decisionLevel() == 0)
        return;

    seen[var(p)] = 1;

    for(int i = trail.size() - 1; i >= trail_lim[0]; i--) {
        Var x = var(trail[i]);
        if(seen[x]) {
            if(reason(x) == CRef_Undef) {
                assert(level(x) > 0);
                out_conflict.push(~trail[i]);
            } else {
                Clause &c = ca[reason(x)];
                //                for (int j = 1; j < c.size(); j++) Minisat (glucose 2.0) loop
                // Bug in case of assumptions due to special data structures for Binary.
                // Many thanks to Sam Bayless (sbayless@cs.ubc.ca) for discover this bug.
                for(int j = ((c.size() == 2) ? 0 : 1); j < c.size(); j++)
                    if(level(var(c[j])) > 0)
                        seen[var(c[j])] = 1;
            }

            seen[x] = 0;
        }
    }

    seen[var(p)] = 0;
}


void Solver::uncheckedEnqueue(Lit p, CRef from) {
    assert(value(p) == l_Undef);
    // if no reason, it's the first at its level, or level is 0
    assert(from != CRef_Undef || decisionLevel() == 0 || trail_lim[decisionLevel() - 1] == trail.size());
    assigns[var(p)] = lbool(!sign(p));
    vardata[var(p)] = mkVarData(from, decisionLevel());
    assert(qhead <= trail.size());
    trail.push_(p);
#ifdef KEEP_IMPL_COUNT
    if (from != CRef_Undef) {
        Clause &cl = ca[from];
        if (cl.learned() || cl.permLearned()) {
            learnedPermLearnedImplying++;
        }
    }
#endif
#ifdef DEBUG
    if (from != CRef_Undef) {
        bool foundOneAtDecLev = false;
        Clause &cl = ca[from];
        for (int i = 0; i < cl.size(); i++) {
            assert(cl[i] == p || value(cl[i]) == l_False);
            if (cl[i] != p && level(var(cl[i])) == decisionLevel()) {
                foundOneAtDecLev = true;
            }
        }
        if (!foundOneAtDecLev) {
            printf("dec level is %d\n", decisionLevel());
            for (int i = 0; i < cl.size(); i++) {
                if (cl[i] != p) {
                    printf("level is %d\n", level(var(cl[i])));
                }
            }
        }
        assert(foundOneAtDecLev);
    }
#endif
}


void Solver::bumpForceUNSAT(Lit q) {
    forceUNSAT[var(q)] = sign(q) ? -1 : +1;
    return;
}

void Solver::checkWatchesAreCorrect(Lit l) {
    vec <Watcher> &ws = watches[l];
    for (int i = 0; i < ws.size(); i++) {
        CRef cref = ws[i].cref;
        Clause &cl = ca[cref];
        assert(cl.hasLit(~l));
    }
}

/*_________________________________________________________________________________________________
|
|  propagate : [void]  ->  [Clause*]
|  
|  Description:
|    Propagates all enqueued facts. If a conflict arises, the conflicting clause is returned,
|    otherwise CRef_Undef.
|  
|    Post-conditions:
|      * the propagation queue is empty, even if there was a conflict.
|________________________________________________________________________________________________@*/
CRef Solver::propagate() {
    CRef confl = CRef_Undef;
    int num_props = 0;
    watches.cleanAll();
    watchesBin.cleanAll();
    while(qhead < trail.size()) {
        Lit p = trail[qhead++]; // 'p' is enqueued fact to propagate.
        // This assumption may be wrong if the clause was just imported. ie when the parallel solver
        // imports a clause, it doesn't check if it would imply something
        // ASSERT_OP(level(var(p)), ==, decisionLevel());
        vec <Watcher> &ws = watches[p];
        Watcher *i, *j, *end;
        num_props++;


        // First, Propagate binary clauses
        vec <Watcher> &wbin = watchesBin[p];
        for(int k = 0; k < wbin.size(); k++) {

            Lit imp = wbin[k].blocker;

            if(value(imp) == l_False) {
                addPropagations(num_props);
                return wbin[k].cref;
            }

            if(value(imp) == l_Undef) {
                uncheckedEnqueue(imp, wbin[k].cref);
            }
        }

        // Now propagate other 2-watched clauses
        for(i = j = ws.getData(), end = i + ws.size(); i != end;) {
            // Try to avoid inspecting the clause:
            Lit blocker = i->blocker;
            if(value(blocker) == l_True) {
                *j++ = *i++;
                continue;
            }

            // Make sure the false literal is data[1]:
            CRef cr = i->cref;
            Clause &c = ca[cr];
            Lit false_lit = ~p;
            if(c[0] == false_lit)
                c[0] = c[1], c[1] = false_lit;
            assert(c[1] == false_lit);
            i++;

            // If 0th watch is true, then clause is already satisfied.
            Lit first = c[0];
            Watcher w = Watcher(cr, first);
            if(first != blocker && value(first) == l_True) {

                *j++ = w;
                continue;
            }
#ifdef INCREMENTAL
            if(incremental) { // ----------------- INCREMENTAL MODE
              int choosenPos = -1;
              for (int k = 2; k < c.size(); k++) {

            if (value(c[k]) != l_False){
              if(decisionLevel()>assumptions.size()) {
                choosenPos = k;
                break;
              } else {
                choosenPos = k;

                if(value(c[k])==l_True || !isSelector(var(c[k]))) {
                  break;
                }
              }

            }
              }
              if(choosenPos!=-1) {
            c[1] = c[choosenPos]; c[choosenPos] = false_lit;
            watches[~c[1]].push(w);
            goto NextClause; }
            } else {  // ----------------- DEFAULT  MODE (NOT INCREMENTAL)
#endif
            for(int k = 2; k < c.size(); k++) {

                if(value(c[k]) != l_False) {
                    c[1] = c[k];
                    c[k] = false_lit;
                    watches[~c[1]].push(w);
                    goto NextClause;
                }
            }
#ifdef INCREMENTAL
            }
#endif
            // Did not find watch -- clause is unit under assignment:
            *j++ = w;
            if(value(first) == l_False) {
                confl = cr;
                qhead = trail.size();
                // Copy the remaining watches:
                while(i < end)
                    *j++ = *i++;
            } else {
                uncheckedEnqueue(first, cr);


            }
            NextClause:;
        }
        ws.shrink(i - j);
    }
    addPropagations(num_props);


#ifdef PRINT_TRAIL_LENGTH
    if (confl != CRef_Undef && decisionLevel() > 1) {
        printf("parent_trail_size: %d\n", trail_lim[decisionLevel() - 1]);
    }
#endif
    return confl;
}

void Solver::addPropagations(int count) {
    stats[propagations] += count;
    simpDB_props -= count;
    // it's possible for count * totalClauseCount to overflow int
    stats[clCountTimesProps] += (long) count * totalClauseCount();
}

void Solver::maybeReduceDB() {
    if((!glureduce && learned.size() > firstReduceDB) ||
       (glureduce && conflicts >= ((unsigned int) curRestart * nbclausesbeforereduce))) {

        if(learned.size() > 0) {
            curRestart = (conflicts / nbclausesbeforereduce) + 1;
            reduceDB();
            if(!panicModeIsEnabled())
                nbclausesbeforereduce += incReduceDB;
        }
    }
}

/*_________________________________________________________________________________________________
|
|  reduceDB : ()  ->  [void]
|  
|  Description:
|    Remove half of the learned clauses, minus the clauses locked by the current assignment. Locked
|    clauses are clauses that are reason to some assignment. Binary clauses are never removed.
|________________________________________________________________________________________________@*/
void Solver::reduceDB() {
    int i, j;
    stats[reduceDb]++;
    if (!compareLbd)
        sort(learned, reduceDBAct_lt(ca));
    else {
        sort(learned, reduceDB_lt(ca));

        // We have a lot of "good" clauses, it is difficult to compare them. Keep more !
        if(ca[learned[learned.size() / RATIOREMOVECLAUSES]].lbd() <= 3) nbclausesbeforereduce += specialIncReduceDB;
        // Useless :-)
        if(ca[learned.last()].lbd() <= 5) nbclausesbeforereduce += specialIncReduceDB;

    }
    // Don't delete binary or locked clauses. From the rest, delete clauses from the first half
    // Keep clauses which seem to be usefull (their lbd was reduce during this sequence)

    int limit = learned.size() / 2;

    for(i = j = 0; i < learned.size(); i++) {
        Clause &c = ca[learned[i]];
        if (canRemoveLearnedClause(c) && (i < limit)) {
            removeClause(learned[i]);
        }
        else {
            if(!c.canBeDel()) limit++; //we keep c, so we can delete an other clause
            c.setCanBeDel(true);       // At the next step, c can be deleted
            learned[j++] = learned[i];
        }
    }
    learned.shrink(i - j);
    checkGarbage();
}

bool Solver::canRemoveLearnedClause(Clause &c) {
    return c.lbd() > 2 && c.size() > 2 && c.canBeDel() && !locked(c);
}

void Solver::removeSatisfied(vec <CRef> &cs) {

    int i, j;
    for(i = j = 0; i < cs.size(); i++) {
        Clause &c = ca[cs[i]];
        if(satisfied(c))
            removeClause(cs[i]);
        else
            cs[j++] = cs[i];
    }
    cs.shrink(i - j);
}


void Solver::rebuildOrderHeap() {
    vec <Var> vs;
    for(Var v = 0; v < nVars(); v++)
        if(decision[v] && value(v) == l_Undef)
            vs.push(v);
    order_heap.build(vs);

}


/*_________________________________________________________________________________________________
|
|  simplify : [void]  ->  [bool]
|  
|  Description:
|    Simplify the clause database according to the current top-level assigment. Currently, the only
|    thing done here is the removal of satisfied clauses, but more things can be put here.
|________________________________________________________________________________________________@*/
bool Solver::simplify() {
    assert(decisionLevel() == 0);
    if(!ok) return ok = false;
    else {
        CRef cr = propagate();
        if(cr != CRef_Undef) {
            return ok = false;
        }
    }


    if(nAssigns() == simpDB_assigns || (simpDB_props > 0))
        return true;

    // Remove satisfied clauses:
    removeSatisfied(learned);
    removeSatisfied(permanentlyLearned);
    if(remove_satisfied) // Can be turned off.
        removeSatisfied(clauses);
    checkGarbage();
    rebuildOrderHeap();

    simpDB_assigns = nAssigns();
    simpDB_props = stats[clauseLiterals] + stats[learnedLiterals]; // (shouldn't depend on stats really, but it will do for now)
    return true;
}


void Solver::adaptSolver() {
    bool adjusted = false;
    bool reinit = false;
    LOG(logger, 2, "c\nc Solver " << cpuThreadId << ": try to adapt solver strategies at " << realTimeSecSinceStart() << ". There are " << learned.size() << " learned clauses and " << permanentlyLearned.size() << " permanently learned\n, " << stats[removedClauses] << " clauses were removed\n");
    /*  printf("c Adjusting solver for the SAT Race 2015 (alpha feature)\n");
    printf("c key successive Conflicts       : %" PRIu64"\n",stats[noDecisionConflict]);
    printf("c nb unary clauses learned        : %" PRIu64"\n",stats[nbUn]);
    printf("c key avg dec per conflicts      : %.2f\n", (float)decisions / (float)conflicts);*/
    float decpc = (float) stats[decisions] / (float) conflicts;
    // If we can't perm learn low lbd, doing this change would make little sense
    if(decpc <= 1.2 && mayPermLearnLowLbd) {
        coLBDBound = 4;
        compareLbd = false;
        glureduce = true;
        adjusted = true;
        LOG(logger, 2, "c Adjusting for low decision levels.\n");
        reinit = true;
        firstReduceDB = 2000;
        nbclausesbeforereduce = firstReduceDB;
        curRestart = (conflicts / nbclausesbeforereduce) + 1;
        incReduceDB = 0;
    }
    if(stats[noDecisionConflict] < 30000) {
        luby_restart = true;
        luby_restart_factor = 100;

        var_decay = 0.999;
        max_var_decay = 0.999;
        adjusted = true;
        LOG(logger, 2, "c Adjusting for low successive conflicts.\n");
    }
    if(stats[noDecisionConflict] > 54400 && mayPermLearnLowLbd) {
        LOG(logger, 2, "c Adjusting for high successive conflicts.\n");
        coLBDBound = 3;
        compareLbd = false;
        glureduce = true;
        firstReduceDB = 30000;
        var_decay = 0.99;
        max_var_decay = 0.99;
        randomize_on_restarts = 1;
        adjusted = true;
    }
    if(stats[DL2] - stats[binaryClauses] > 20000) {
        var_decay = 0.91;
        max_var_decay = 0.91;
        adjusted = true;
        LOG(logger, 2, "c Adjusting for a very large number of true glue clauses found.\n");
    }
    if(!adjusted) {
        LOG(logger, 2, "c Nothing extreme in this problem, continue with glucose default strategies.\n");
    }
    LOG(logger, 2, "c\n");
    if(adjusted) { // Let's reinitialize the glucose restart strategy counters
        lbdQueue.fastclear();
        sumLBD = 0;
        conflictsRestarts = 0;
    }

    if(adjusted) {
        int moved = 0;
        int i, j;
        for(i = j = 0; i < learned.size(); i++) {
            Clause &c = ca[learned[i]];
            if((int) c.lbd() <= coLBDBound) {
                updateStatsForClauseChanged(c, -1);
                permanentlyLearned.push(learned[i]);
                c.setPermLearned();
                updateStatsForClauseChanged(c, 1);
                moved++;
            }
            else {
                learned[j++] = learned[i];
            }
        }
        learned.shrink(i - j);
        LOG(logger, 2, "c Activating Chanseok Strategy: moved " << moved << " clauses to the permanent set.\n");
    }

    if(reinit) {
        assert(decisionLevel() == 0);
        for(int i = 0; i < learned.size(); i++) {
            removeClause(learned[i]);
        }
        learned.shrink(learned.size());
        checkGarbage();
/*
    order_heap.clear();
    for(int i=0;i<nVars();i++) {
        polarity[i] = true; 
        activity[i]=0.0;
        if (decision[i]) order_heap.insert(i);
    }
    printf("c reinitialization of all variables activity/phase/learned clauses.\n");
*/
        LOG(logger, 2, "c Removing of non permanent clauses.\n");
    }
}

bool Solver::propagateAndMaybeLearnFromConflict(bool &foundEmptyClause, bool &blocked, vec<Lit> &learned_clause, vec<Lit> &selectors) {
    CRef confl = propagateAlsoGpu(foundEmptyClause);
    if (foundEmptyClause) {
        return true;
    }
    if (confl == CRef_Undef) {
        tryCopyTrailForGpu(decisionLevel());
        return false;
    }
    newDescent = false;
    stats[sumDecisionLevels] += decisionLevel();
    stats[sumTrail] += trail.size();
    // CONFLICT
    conflicts++;
    conflictsRestarts++;

    if(decisionLevel() == 0) {
        // conflict at decision level 0 means that we've found the empty clause
        foundEmptyClause = true;
        return true;
    }

    trailQueue.push(trail.size());
    // BLOCK RESTART (CP 2012 paper)
    if(conflictsRestarts > LOWER_BOUND_FOR_BLOCKING_RESTART && lbdQueue.isvalid() && trail.size() > R * trailQueue.getavg()) {
        lbdQueue.fastclear();
        stats[stopsRestarts]++;
        if(!blocked) {
            stats[lastBlockAtRestart] = starts;
            stats[stopsRestartsSame]++;
            blocked = true;
        }
    }
    learned_clause.clear();
    selectors.clear();
    int backtrack_level;
    unsigned int nblevels;
    unsigned int szWithoutSelectors;
    analyze(confl, learned_clause, selectors, backtrack_level, nblevels, szWithoutSelectors);
    sendClauseToGpu(learned_clause, nblevels);
#ifdef DEBUG
    if (learned_clause.size() > 1) {
        assert(level(var(learned_clause[0])) >= decisionLevel());
        assert(backtrack_level < decisionLevel());
    }
#endif
    lbdQueue.push(nblevels);
    sumLBD += nblevels;

    cancelUntil(backtrack_level);

    if(certifiedUNSAT) {
        if(vbyte) {
            write_char('a');
            for(int i = 0; i < learned_clause.size(); i++)
                write_lit(2 * (var(learned_clause[i]) + 1) + sign(learned_clause[i]));
            write_lit(0);
        }
        else {
            for(int i = 0; i < learned_clause.size(); i++)
                fprintf(certifiedOutput, "%i ", (var(learned_clause[i]) + 1) *
                                                (-2 * sign(learned_clause[i]) + 1));
            fprintf(certifiedOutput, "0\n");
        }
    }
    if(learned_clause.size() == 1) {
        uncheckedEnqueue(learned_clause[0]);
        stats[unaryClauses]++;
    } else {
        CRef cr;
        cr = learnClause(learned_clause, false, nblevels);
#ifdef INCREMENTAL
        ca[cr].setSizeWithoutSelectors(szWithoutSelectors);
#endif
        uncheckedEnqueue(learned_clause[0], cr);
    }
    varDecayActivity();
    claDecayActivity();
    return true;
}

bool Solver::tryCopyTrailForGpu(int level) {
    return false;
}

void Solver::sendClauseToGpu(vec<Lit> &lits, int lbd) {
}

/*_________________________________________________________________________________________________
|
|  search : (nof_conflicts : int) (params : const SearchParams&)  ->  [lbool]
|  
|  Description:
|    Search for a model the specified number of conflicts. 
|    NOTE! Use negative value for 'nof_conflicts' indicate infinity.
|  
|  Output:
|    'l_True' if a partial assigment that is consistent with respect to the clauseset is found. If
|    all variables are decision variables, this means that the clause set is satisfiable. 'l_False'
|    if the clause set is unsatisfiable. 'l_Undef' if the bound on number of conflicts is reached.
|________________________________________________________________________________________________@*/
lbool Solver::search(int nof_conflicts) {
    assert(ok);
    int conflictC = 0;
    vec <Lit> learned_clause, selectors;
    bool blocked = false;
    bool aDecisionWasMade = false;

    stats[starts]++;
    for(; ;) {
        if (finisher.shouldIStop(cpuThreadId)) {
            return l_Undef;
        }
        bool foundEmptyClause;
        bool foundConflict = propagateAndMaybeLearnFromConflict(foundEmptyClause, blocked, learned_clause, selectors);

        if (foundConflict) {
            if (foundEmptyClause) return l_False;

            if (conflicts % 5000 == 0 && var_decay < max_var_decay) var_decay += 0.01;
            if (verbosity() >= 1 && starts>0 && verb.everyConflicts > 0 && conflicts % verb.everyConflicts == 0) {
                JsonWriter wr(std::cout);
                printEncapsulatedStats(wr, std::cout);
            }
            if(adaptStrategies && conflicts == 100000) {
                cancelUntil(0);
                adaptSolver();
                adaptStrategies = false;
                return l_Undef;
            }
            if (!aDecisionWasMade)
                stats[noDecisionConflict]++;
            aDecisionWasMade = false;
            conflictC++;
        }
        else  {
            // Our dynamic restart, see the SAT09 competition compagnion paper
            if((luby_restart && nof_conflicts <= conflictC) ||
               (!luby_restart && (lbdQueue.isvalid() && ((lbdQueue.getavg() * K) > (sumLBD / conflictsRestarts))))) {
                lbdQueue.fastclear();
                progress_estimate = progressEstimate();
                int bt = 0;
#ifdef INCREMENTAL
                if(incremental) // DO NOT BACKTRACK UNTIL 0.. USELESS
                    bt = (decisionLevel()<assumptions.size()) ? decisionLevel() : assumptions.size();
#endif
                newDescent = true;

                if(randomize_on_restarts || fixed_randomize_on_restarts) {
                    randomDescentAssignments = (uint32_t) drand(random_seed);
                }

                cancelUntil(bt);
                return l_Undef;
            }

            // Simplify the set of problem clauses:
            if(decisionLevel() == 0 && !simplify()) {
                return l_False;
            }
            // Perform clause database reduction !
            maybeReduceDB();
            Lit next = lit_Undef;
            while(decisionLevel() < assumptions.size()) {
                // Perform user provided assumption:
                Lit p = assumptions[decisionLevel()];
                if(value(p) == l_True) {
                    // Dummy decision level:
                    newDecisionLevel();
                } else if(value(p) == l_False) {
                    analyzeFinal(~p, conflict);
                    return l_False;
                    next = p;
                    break;
                }
            }

            if(next == lit_Undef) {
                // New variable decision:
                stats[decisions]++;
                next = pickBranchLit();
                if(next == lit_Undef) {
                    LOG(logger, 2, "c last restart ## conflicts  :   " << conflictC << " " << decisionLevel());
                    // Model found:
                    return l_True;
                }
            }

            // Increase decision level and enqueue 'next'
            aDecisionWasMade = true;
            newDecisionLevel();
            uncheckedEnqueue(next);
        }
    }
}

CRef Solver::learnClause(vec<Lit> &lits, bool fromGpu, int nblevels) {
    CRef cr;
    assert(lits.size() >= 2);
    if(nblevels <= coLBDBound) {
        cr = ca.alloc(lits, false, fromGpu, true);
        permanentlyLearned.push(cr);
    } else {
        cr = ca.alloc(lits, true, fromGpu, false);
        learned.push(cr);
        claBumpActivity(ca[cr]);
    }
    updateStatsForClauseChanged(ca[cr], 1);
    ca[cr].setLBD(nblevels);
#ifdef PRINT_DETAILS_CLAUSES
    {
        SyncOut so;
        std::cout << "learn: thread " << cpuThreadId << " " << ca[cr] << std::endl;
    }
#endif
    attachClause(cr);
    if(nblevels <= 2) { stats[DL2]++; } // stats
    if(ca[cr].size() == 2) stats[binaryClauses]++; // stats
    return cr;
}

// Returns a clause found to be in conflict
// foundEmptyClause: if we found an empty clause
// This method must ensure that between two times when we
// copy vars for the gpu: we've imported gpu clauses
CRef Solver::propagateAlsoGpu(bool &foundEmptyClause) {

    CRef confl = gpuImportClauses(foundEmptyClause);
    if (foundEmptyClause) {
        return CRef_Undef;
    }
    if (confl != CRef_Undef) {
        return confl;
    }
    confl = propagate();
    return confl;
}

double Solver::progressEstimate() const {
    // I've simplified this because the previous version wasn't thread safe, this was called
    // from one thread while another thread would update the values at the same time
    return (float)stats[atLevel0] / nVars();
}


// NOTE: assumptions passed in member-variable 'assumptions'.

lbool Solver::solve_(bool do_simp, bool turn_off_simp) // Parameters are useless in core but useful for SimpSolver....
{

    if(incremental && certifiedUNSAT) {
        printf("Can not use incremental and certified unsat in the same time\n");
        exit(-1);
    }

    model.clear();
    conflict.clear();
    if(!ok) return l_False;
    double curCpuTime = cpuTimeSec();

    stats[solves]++;


    lbool status = l_Undef;
    // this code isn't used in multithreaded, so it doesn't really need to use logger
    if(!incremental && verbosity() >= 1) {
        printf("c ========================================[ MAGIC CONSTANTS ]==============================================\n");
        printf("c | Constants are supposed to work well together :-)                                                      |\n");
        printf("c | however, if you find better choices, please let us known...                                           |\n");
        printf("c |-------------------------------------------------------------------------------------------------------|\n");
        if(adaptStrategies) {
            printf("c | Adapt dynamically the solver after 100000 conflicts (restarts, reduction strategies...)               |\n");
            printf("c |-------------------------------------------------------------------------------------------------------|\n");
        }
        printf("c |                                |                                |                                     |\n");
        printf("c | - Restarts:                    | - Reduce Clause DB:            | - Minimize Asserting:               |\n");
        if(coLBDBound > 2) {
            printf("c |   * LBD Queue    : %6d      |     chanseok Strategy          |    * size < %3d                     |\n", lbdQueue.maxSize(),
                   lbSizeMinimizingClause);
            printf("c |   * Trail  Queue : %6d      |   * learned size     : %6d  |    * lbd  < %3d                     |\n", trailQueue.maxSize(),
                   firstReduceDB, lbLBDMinimizingClause);
            printf("c |   * K            : %6.2f      |   * Bound LBD   : %6d       |                                     |\n", K, coLBDBound);
            printf("c |   * R            : %6.2f      |   * Protected :  (lbd)< %2d     |                                     |\n", R, lbLBDFrozenClause);
        } else {
            printf("c |   * LBD Queue    : %6d      |   * First     : %6f         |    * size < %3d                     |\n", lbdQueue.maxSize(),
                   nbclausesbeforereduce, lbSizeMinimizingClause);
            printf("c |   * Trail  Queue : %6d      |   * Inc       : %6d         |    * lbd  < %3d                     |\n", trailQueue.maxSize(), incReduceDB,
                   lbLBDMinimizingClause);
            printf("c |   * K            : %6.2f      |   * Special   : %6d         |                                     |\n", K, specialIncReduceDB);
            printf("c |   * R            : %6.2f      |   * Protected :  (lbd)< %2d     |                                     |\n", R, lbLBDFrozenClause);
        }
        printf("c |                                |                                |                                     |\n");
        printf("c ==================================[ Search Statistics (every %6d conflicts) ]=========================\n", verb.everyConflicts);
    }

    // Search:
    int curr_restarts = 0;
    while(status == l_Undef) {
        status = search(
                luby_restart ? luby(restart_inc, curr_restarts) * luby_restart_factor : 0); // the parameter is useless in glucose, kept to allow modifications

        if(!withinBudget() || finisher.shouldIStop(cpuThreadId)) break;
        curr_restarts++;
    }

    if(certifiedUNSAT) { // Want certified output
        if(status == l_False) {
            if(vbyte) {
                write_char('a');
                write_lit(0);
            }
            else {
                fprintf(certifiedOutput, "0\n");
            }
        }
        fclose(certifiedOutput);
    }


    if(status == l_True) {
        // Extend & copy model:
        model.growTo(nVars());
        for(int i = 0; i < nVars(); i++) model[i] = value(i);
    } else if(status == l_False && conflict.size() == 0)
        ok = false;


    cancelUntil(0);


    double finalCpuTime = cpuTimeSec();
    if(status == l_True) {
        nbSatCalls++;
        totalTime4Sat += (finalCpuTime - curCpuTime);
    }
    if(status == l_False) {
        nbUnsatCalls++;
        totalTime4Unsat += (finalCpuTime - curCpuTime);
    }


    return status;

}





//=================================================================================================
// Writing CNF to DIMACS:
// 
// FIXME: this needs to be rewritten completely.

static Var mapVar(Var x, vec <Var> &map, Var &max) {
    if(map.size() <= x || map[x] == -1) {
        map.growTo(x + 1, -1);
        map[x] = max++;
    }
    return map[x];
}


void Solver::toDimacs(FILE *f, Clause &c, vec <Var> &map, Var &max) {
    if(satisfied(c)) return;

    for(int i = 0; i < c.size(); i++)
        if(value(c[i]) != l_False)
            fprintf(f, "%s%d ", sign(c[i]) ? "-" : "", mapVar(var(c[i]), map, max) + 1);
    fprintf(f, "0\n");
}


void Solver::toDimacs(const char *file, const vec <Lit> &assumps) {
    FILE *f = fopen(file, "wr");
    if(f == NULL)
        fprintf(stderr, "could not open file %s\n", file), exit(1);
    toDimacs(f, assumps);
    fclose(f);
}


void Solver::toDimacs(FILE *f, const vec <Lit> &assumps) {
    // Handle case when solver is in contradictory state:
    if(!ok) {
        fprintf(f, "p cnf 1 2\n1 0\n-1 0\n");
        return;
    }

    vec <Var> map;
    Var max = 0;

    // Cannot use removeClauses here because it is not safe
    // to deallocate them at this point. Could be improved.
    int cnt = 0;
    for(int i = 0; i < clauses.size(); i++)
        if(!satisfied(ca[clauses[i]]))
            cnt++;

    for(int i = 0; i < clauses.size(); i++)
        if(!satisfied(ca[clauses[i]])) {
            Clause &c = ca[clauses[i]];
            for(int j = 0; j < c.size(); j++)
                if(value(c[j]) != l_False)
                    mapVar(var(c[j]), map, max);
        }

    // Assumptions are added as unit clauses:
    cnt += assumptions.size();

    fprintf(f, "p cnf %d %d\n", max, cnt);

    for(int i = 0; i < assumptions.size(); i++) {
        assert(value(assumptions[i]) != l_False);
        fprintf(f, "%s%d 0\n", sign(assumptions[i]) ? "-" : "", mapVar(var(assumptions[i]), map, max) + 1);
    }

    for(int i = 0; i < clauses.size(); i++)
        toDimacs(f, ca[clauses[i]], map, max);

    if(verbosity() > 0)
        printf("Wrote %d clauses with %d variables.\n", cnt, max);
}


//=================================================================================================
// Garbage Collection methods:

void Solver::relocAll(ClauseAllocator &to) {
    // All watchers:
    // for (int i = 0; i < watches.size(); i++)
    watches.cleanAll();
    watchesBin.cleanAll();
    for(int v = 0; v < nVars(); v++)
        for(int s = 0; s < 2; s++) {
            Lit p = mkLit(v, s);
            // printf(" >>> RELOCING: %s%d\n", sign(p)?"-":"", var(p)+1);
            vec <Watcher> &ws = watches[p];
            for(int j = 0; j < ws.size(); j++)
                ca.reloc(ws[j].cref, to);
            vec <Watcher> &ws2 = watchesBin[p];
            for(int j = 0; j < ws2.size(); j++)
                ca.reloc(ws2[j].cref, to);
        }

    // All reasons:
    //
    for(int i = 0; i < trail.size(); i++) {
        Var v = var(trail[i]);

        if(reason(v) != CRef_Undef && (ca[reason(v)].reloced() || locked(ca[reason(v)])))
            ca.reloc(vardata[v].reason, to);
    }

    // All learned:
    //
    for(int i = 0; i < learned.size(); i++)
        ca.reloc(learned[i], to);

    for(int i = 0; i < permanentlyLearned.size(); i++)
        ca.reloc(permanentlyLearned[i], to);

    // All original:
    //
    for(int i = 0; i < clauses.size(); i++)
        ca.reloc(clauses[i], to);
}


void Solver::garbageCollect() {
    // Initialize the next region to a size corresponding to the estimated utilization degree. This
    // is not precise but should avoid some unnecessary reallocations for the new region:
    ClauseAllocator to(ca.size() - ca.wasted());
    relocAll(to);
    LOG(logger, 2, "c  Garbage collection:   " << ca.size() * ClauseAllocator::Unit_Size << " bytes => " << to.size() * ClauseAllocator::Unit_Size << "\n");
    to.moveTo(ca);
}

//--------------------------------------------------------------
// Functions related to MultiThread.
// Useless in case of single core solver (aka original glucose)
// Keep them empty if you just use core solver
//--------------------------------------------------------------

bool Solver::panicModeIsEnabled() {
    return false;
}

CRef Solver::gpuImportClauses(bool &foundEmptyClause) {
    foundEmptyClause = false;
    return CRef_Undef;
}

void Solver::printStats(JsonWriter &jw) {
    for (auto const& e : statNames) {
        jw.write(e.second.c_str(), stats[e.first]);
    }
    jw.write("original", (unsigned long) clauses.size());
    jw.write("conflicts", conflicts);
    jw.write("nbClausesBeforeReduce", (unsigned long) nbclausesbeforereduce);
}

void Solver::printEncapsulatedStats(JsonWriter &writer, std::ostream &ost) {
    JStats jstats(writer, ost);
    printStats(writer);
}

double Glucose::luby(double y, int x) {
    // Find the finite subsequence that contains index 'x', and the
    // size of that subsequence:
    int size, seq;
    for(size = 1, seq = 0; size < x + 1; seq++, size = 2 * size + 1);

    while(size - 1 != x) {
        size = (size - 1) >> 1;
        seq--;
        x = x % size;
    }
    return pow(y, seq);
}

