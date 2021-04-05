/***************************************************************************************[Solver.h]
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

#ifndef Glucose_Solver_h
#define Glucose_Solver_h

/*
#include "../mtl/Clone.h"
#include "../mtl/Heap.h"
#include "../mtl/Vec.h"
#include "BoundedQueue.h"
#include "SolverTypes.h"
*/

// #define KEEP_IMPL_COUNT

#include "mtl/Heap.h"
#include "mtl/Alg.h"
#include "utils/Options.h"
#include "satUtils/SolverTypes.h"
#include "core/BoundedQueue.h"
#include "core/Constants.h"
#include "mtl/Clone.h"
#include "utils/JsonWriter.h"
#include <map>
#include <string>

namespace GpuShare {
    class Logger;
}

namespace Glucose {
// Core stats 

enum CoreStats {
#define X(v) v,
#include "core/CoreSolverStats.h"
#undef X
} ;

template <class S> float getStatAvg(vec<S*>& solvers, CoreStats stat) {
    return (float) getStatSum(solvers, stat) / solvers.size();
}

struct Finisher;

//=================================================================================================
// Solver -- the main class:

class Solver : public Clone {

    // So that they can set solver's configuration
    friend class SolverConfiguration;
    friend class GpuMultiSolver;

public:

    // Constructor/Destructor:
    Solver(int cpuThreadId, Finisher &_finisher, const GpuShare::Logger &logger);
    Solver(const  Solver &s, int cpuThreadId);

    virtual ~Solver();
    
    /**
     * Clone function
     */
    virtual Clone* clone(int threadId) const {
        return  new Solver(*this, threadId);
    }   

    // Problem specification:
    //
    virtual Var     newVar    (bool polarity = true, bool dvar = true); // Add a new variable with parameters specifying variable mode.
    bool    addClause (const vec<Lit>& ps);                     // Add a clause to the solver. 
    bool    addEmptyClause();                                   // Add the empty clause, making the solver contradictory.
    bool    addClause (Lit p);                                  // Add a unit clause to the solver. 
    bool    addClause (Lit p, Lit q);                           // Add a binaryClausesary clause to the solver. 
    bool    addClause (Lit p, Lit q, Lit r);                    // Add a ternary clause to the solver. 
    virtual bool    addClause_(      vec<Lit>& ps);                     // Add a clause to the solver without making superflous internal copy. Will
                                                                // change the passed vector 'ps'.
    // Solving:
    //
    bool    simplify     ();                        // Removes already satisfied clauses.
    bool    solve        (const vec<Lit>& assumps); // Search for a model that respects a given set of assumptions.
    lbool   solveLimited (const vec<Lit>& assumps); // Search for a model that respects a given set of assumptions (With resource constraints).
    bool    solve        ();                        // Search without assumptions.
    bool    solve        (Lit p);                   // Search for a model that respects a single assumption.
    bool    solve        (Lit p, Lit q);            // Search for a model that respects two assumptions.
    bool    solve        (Lit p, Lit q, Lit r);     // Search for a model that respects three assumptions.

    bool    okay         () const;                  // FALSE means solver is in a conflicting state

    CRef    propagate    ();                        // Perform unit propagation. Returns possibly conflicting clause. Visible for testing

    int     level        (Var x) const;             // visible for testing

    virtual void    cancelUntil (int level);                // Backtrack until a certain level. Visible for testing

    void    newDecisionLevel ();                    // Begins a new decision level. Visible for testing

    void    uncheckedEnqueue (Lit p, CRef from = CRef_Undef);        // Enqueue a literal. Assumes value of literal is undefined.
    bool    enqueue          (Lit p, CRef from = CRef_Undef);        // Test if fact 'p' contradicts current state, return false if it does, enqueue otherwise.

       // Convenience versions of 'toDimacs()':
    void    toDimacs     (FILE* f, const vec<Lit>& assumps);            // Write CNF to file in DIMACS-format.
    void    toDimacs     (const char *file, const vec<Lit>& assumps);
    void    toDimacs     (FILE* f, Clause& c, vec<Var>& map, Var& max);
    void    toDimacs     (const char* file);
    void    toDimacs     (const char* file, Lit p);
    void    toDimacs     (const char* file, Lit p, Lit q);
    void    toDimacs     (const char* file, Lit p, Lit q, Lit r);
 
    // Display clauses and literals
    void printLit(Lit l);
    void printClause(CRef c);
    void printInitialClause(CRef c);
    
    // Variable mode:
    // 
    void    setPolarity    (Var v, bool b); // Declare which polarity the decision heuristic should use for a variable. Requires mode 'polarity_user'.
    void    setDecisionVar (Var v, bool b); // Declare if a variable should be eligible for selection in the decision heuristic.

    // Read state:
    //
    lbool   value      (Var x) const;       // The current value of a variable.
    lbool   value      (Lit p) const;       // The current value of a literal.
    lbool   modelValue (Var x) const;       // The value of a variable in the last model. The last call to solve must have been satisfiable.
    lbool   modelValue (Lit p) const;       // The value of a literal in the last model. The last call to solve must have been satisfiable.
    int     nAssigns   ()      const;       // The current number of assigned literals.
    int     nClauses   ()      const;       // The current number of original clauses.
    int     nLearned   ()      const;       // The current number of learned clauses.
    int     totalClauseCount() const;       // Total number of clauses
    int     nVars      ()      const;       // The current number of variables.
    int     nFreeVars  ()      ;

    inline char valuePhase(Var v) {return polarity[v];}

    // Incremental mode
    void setIncrementalMode();
    void initNbInitialVars(int nb);
    bool isIncremental();
    // Resource contraints:
    //
    void    setConfBudget(int64_t x);
    void    setPropBudget(int64_t x);
    void    budgetOff();

    // Memory managment:
    //
    virtual void garbageCollect();
    void    checkGarbage(double gf);
    void    checkGarbage();

    void setVerbosity(Verbosity v) {verb = v;}
    Verbosity getVerbosity() {return verb; }

    int verbosity() { return verb.global; }

    Lit pickBranchLit(); // Return the next decision variable.

    // Extra results: (read-only member variable)
    //
    vec<lbool> model;             // If problem is satisfiable, this vector contains the model (if any).
    vec<Lit>   conflict;          // If problem is unsatisfiable (possibly under assumptions),
                                  // this vector represent the final conflict clause expressed in the assumptions.
    
    // Constants For restarts
    double    K;
    double    R;
    double    sizeLBDQueue;
    double    sizeTrailQueue;

    // Constants for reduce DB
    int          firstReduceDB;
    int          incReduceDB;
    int          specialIncReduceDB;
    unsigned int lbLBDFrozenClause;
    // If true: when adjusting the solver, may decide to permanently learn clauses with an lbd > 2
    bool         mayPermLearnLowLbd;
    bool         compareLbd; // if false: compare activities to tell which clauses to remove
    double       minactFactor; // If not 0, use this factor to choose which clauses to keep, don't compare them to each other
    int          coLBDBound; // Keep all learned with lbd<=coLBDBound, used if permLearnLowLbd
    // Constant for reducing clause
    int          lbSizeMinimizingClause;
    unsigned int lbLBDMinimizingClause;

    // Constant for heuristic
    double    var_decay;
    double    max_var_decay;
    double    clause_decay;
    double    random_var_freq;
    double    random_seed;
    int       ccmin_mode;         // Controls conflict clause minimization (0=none, 1=basic, 2=deep).
    int       phase_saving;       // Controls the level of phase saving (0=none, 1=limited, 2=full).
    bool      rnd_pol;            // Use random polarities for branching heuristics.
    bool      rnd_init_act;       // Initialize variable activities with a small random value.
    bool      randomizeFirstDescent; // the first decisions (until first cnflict) are made randomly
                                     // Useful for syrup!
    
    // Constant for Memory managment
    double    garbage_frac;       // The fraction of wasted memory allowed before a garbage collection is triggered.

    // Certified UNSAT ( Thanks to Marijn Heule
    // New in 2016 : proof in DRAT format, possibility to use binary output
    FILE*               certifiedOutput;
    bool                certifiedUNSAT;
    bool                vbyte;

    void write_char (unsigned char c);
    void write_lit (int n);


    // Panic mode. 
    // Save memory
    uint32_t panicModeLastRemoved, panicModeLastRemovedShared;
    
    virtual CRef gpuImportClauses(bool &foundEmptyClause);

    virtual bool tryCopyTrailForGpu(int level);
    virtual void sendClauseToGpu(vec<Lit> &lits, int lbd);

    virtual bool panicModeIsEnabled();

    // Statistics 
    vec<uint64_t> stats;
    uint64_t conflictsRestarts;
    uint64_t conflicts;

    int litLevel(Lit p) {return value(p) != l_False ? INT_MAX : level(var(p));}

    inline int decisionLevel() const { return trail_lim.size(); }

    virtual void reduceDB();  // Reduce the set of learned clauses. Visible for testing
    virtual void maybeReduceDB(); // reduce db if necessary

    lbool search(int nof_conflicts);// Search for a given number of conflicts.

    // returns if there was a conflict
    bool propagateAndMaybeLearnFromConflict(bool &foundEmptyClause, bool &blocked, vec<Lit> &learned_clause, vec<Lit> &selectors);

    // for progress estimator
    Lit getFromTrail(int i) {
        return trail[i];
    }

    void printStats(JsonWriter &jsonWriter);

    void printEncapsulatedStats();
    long getApproximateMemAllocated() {return ca.getCap(); }

protected:

    Finisher &finisher;
    // it would be possible to make it a stats, but it's read from many places I think it's easier this way
    std::map<int, std::string> statNames;

    void insertStatNames();

    Verbosity verb;

    long curRestart;

    // Alpha variables
    bool glureduce;
    uint32_t restart_inc;
    bool  luby_restart;
    bool adaptStrategies;
    uint32_t luby_restart_factor;
    bool randomize_on_restarts, fixed_randomize_on_restarts, newDescent;
    uint32_t randomDescentAssignments;
    bool forceUnsatOnNewDescent;

    // Helper structures:
    //
    struct VarData { CRef reason; int level; };
    static inline VarData mkVarData(CRef cr, int l){ VarData d = {cr, l}; return d; }


    struct WatcherDeleted
    {
        const ClauseAllocator& ca;
        WatcherDeleted(const ClauseAllocator& _ca) : ca(_ca) {}
        bool operator()(const Watcher& w) const { return ca[w.cref].mark() == 1; }
    };

    struct VarOrderLt {
        const vec<double>&  activity;
        bool operator () (Var x, Var y) const { return activity[x] > activity[y]; }
        VarOrderLt(const vec<double>&  act) : activity(act) { }
    };


    // Solver state:
    //
    int                lastIndexRed;
    bool                ok;               // If FALSE, the constraints are already unsatisfiable. No part of the solver state may be used!
    double              cla_inc;          // Amount to bump next clause with.
    vec<double>         activity;         // A heuristic measurement of the activity of a variable.
    double              var_inc;          // Amount to bump next variable with.
    OccLists<Lit, vec<Watcher>, WatcherDeleted>
                        watches;          // 'watches[lit]' is a list of constraints watching 'lit' (will go there if literal becomes true).
    OccLists<Lit, vec<Watcher>, WatcherDeleted>
                        watchesBin;          // 'watches[lit]' is a list of constraints watching 'lit' (will go there if literal becomes true).
    vec<CRef>           clauses;          // List of problem clauses.
    vec<CRef>           learned;          // List of learned clauses.
    vec<CRef>           permanentlyLearned; // The list of learned clauses kept permanently

    vec<lbool>          assigns;          // The current assignments.
    vec<char>           polarity;         // The preferred polarity of each variable.
    vec<char>           forceUNSAT;
    void                bumpForceUNSAT(Lit q); // Handles the forces

    vec<char>           decision;         // Declares if a variable is eligible for selection in the decision heuristic.
    vec<Lit>            trail;            // Assignment stack; stores all assigments made in the order they were made.
    vec<int>            nbpos;
    vec<int>            trail_lim;        // Separator indices for different decision levels in 'trail'. For a level l: in trail, it's from trail_lim[l - 1] and trail_lim[l] -1 included
    vec<VarData>        vardata;          // Stores reason and level for each variable.
    int                 qhead;            // Head of queue (as index into the trail -- no more explicit propagation queue in MiniSat).
    int                 simpDB_assigns;   // Number of top-level assignments since last execution of 'simplify()'.
    int64_t             simpDB_props;     // Remaining number of propagations that must be made before next execution of 'simplify()'.
    vec<Lit>            assumptions;      // Current set of assumptions provided to solve by the user.
    Heap<VarOrderLt>    order_heap;       // A priority queue of variables ordered with respect to the variable activity.
    double              progress_estimate;// Set by 'search()'.
    bool                remove_satisfied; // Indicates whether possibly inefficient linear scan for satisfied clauses should be performed in 'simplify'.
    vec<unsigned int>   permDiff;           // permDiff[var] contains the current conflict number... Used to count the number of  LBD
    

    // UPDATEVARACTIVITY trick (see competition'09 companion paper)
    vec<Lit> lastDecisionLevel; 

    ClauseAllocator     ca;

    float nbclausesbeforereduce;            // To know when it is time to reduce clause database
    
    // Used for restart strategies
    bqueue<unsigned int> trailQueue,lbdQueue; // Bounded queues for restarts.
    float sumLBD; // used to compute the global average of LBD. Restarts...
    int sumAssumptions;


    // Temporaries (to reduce allocation overhead). Each variable is prefixed by the method in which it is
    // used, exept 'seen' wich is used in several places.
    //
    vec<char>           seen;
    vec<Lit>            analyze_stack;
    vec<Lit>            analyze_toclear;
    vec<Lit>            add_tmp;
    unsigned int  MYFLAG;

    // Initial reduceDB strategy
    double              max_learned;
    double              learnedize_adjust_confl;
    int                 learnedize_adjust_cnt;

    // Resource contraints:
    //
    int64_t             conflict_budget;    // -1 means no budget.
    int64_t             propagation_budget; // -1 means no budget.

    // Variables added for incremental mode
    int incremental; // Use incremental SAT Solver
    int nbVarsInitialFormula; // nb VAR in formula without assumptions (incremental SAT)
    double totalTime4Sat,totalTime4Unsat;
    int nbSatCalls,nbUnsatCalls;
    vec<int> assumptionPositions,initialPositions;
    int cpuThreadId;

    // The number of learned or perm learned clauses implying something at any given time
    // its value only matters if we have KEEP_IMPL_COUNT
    int learnedPermLearnedImplying;

    const GpuShare::Logger &logger;


    // Main internal methods:
    //
    void     insertVarOrder   (Var x);                                                 // Insert a variable in the decision order priority queue.

    CRef     propagateUnaryWatches(Lit p);                                             // Perform propagation on unary watches of p, can find only conflicts
    void     analyze          (CRef confl, vec<Lit>& out_learned, vec<Lit> & selectors, int& out_btlevel,unsigned int &nblevels,unsigned int &szWithoutSelectors);    // (bt = backtrack)
    void     analyzeFinal     (Lit p, vec<Lit>& out_conflict);                         // COULD THIS BE IMPLEMENTED BY THE ORDINARIY "analyze" BY SOME REASONABLE GENERALIZATION?
    bool     litRedundant     (Lit p, uint32_t abstract_levels);                       // (helper method for 'analyze()')
    virtual lbool    solve_           (bool do_simp = true, bool turn_off_simp = false);                                                      // Main solve method (assumptions given in 'assumptions').
    void     removeSatisfied  (vec<CRef>& cs);                                         // Shrink 'cs' to contain only non-satisfied clauses.
    void     rebuildOrderHeap ();

    void     adaptSolver();                                                            // Adapt solver strategies

    void     addPropagations(int count);

    // Maintaining Variable/Clause activity:
    //
    void     varDecayActivity ();                      // Decay all variables with the specified factor. Implemented by increasing the 'bump' value instead.
    void     varBumpActivity  (Var v, double inc);     // Increase a variable with the current 'bump' value.
    void     varBumpActivity  (Var v);                 // Increase a variable with the current 'bump' value.
    void     claDecayActivity ();                      // Decay all clauses with the specified factor. Implemented by increasing the 'bump' value instead.
    void     claBumpActivity  (Clause& c);             // Increase a clause with the current 'bump' value.

    // Operations on clauses:
    //
    void     attachClause     (CRef cr);               // Attach a clause to watcher lists.
    void     detachClause     (CRef cr, bool strict = false); // Detach a clause to watcher lists.
    void     removeClause     (CRef cr);               // Detach and free a clause.
    bool     locked           (const Clause& c) const; // Returns TRUE if a clause is a reason for some implication in the current state.
    bool     satisfied        (const Clause& c) const; // Returns TRUE if a clause is satisfied in the current state.

    void minimisationWithBinaryResolution(vec<Lit> &out_learned);

    virtual void     relocAll         (ClauseAllocator& to);

    void updateStatsForClauseChanged(Clause &cl, int diff); // diff: 1 if added, -1 if removed
    // Misc:
    //
    uint32_t abstractLevel    (Var x) const; // Used to represent an abstraction of sets of decision levels.
    CRef     reason           (Var x) const;
    double   progressEstimate ()      const; // DELETE THIS ?? IT'S NOT VERY USEFUL ...
    bool     withinBudget     ()      const;
    inline bool isSelector(Var v) {return (incremental && v>nbVarsInitialFormula);}

    void checkConflictClause(CRef confl, const char* pos);

    CRef propagateAlsoGpu(bool &foundEmptyClause);

    // learns a clause, decides to put it in permanently learned or not
    CRef learnClause(vec<Lit> &lits, bool fromGpu, int nblevels);

    bool canRemoveLearnedClause(Clause &c);

    void checkWatchesAreCorrect(Lit l);

    // Static helpers:
    //


    /************************************************************
     * Compute LBD functions
     *************************************************************/
    template <typename T>inline unsigned int computeLBD(const T &lits, int end = -1) {
        int nblevels = 0;
        MYFLAG++;
    #ifdef INCREMENTAL
        if(incremental) { // ----------------- INCREMENTAL MODE
          if(end==-1) end = lits.size();
          int nbDone = 0;
          for(int i=0;i<lits.size();i++) {
            if(nbDone>=end) break;
            if(isSelector(var(lits[i]))) continue;
            nbDone++;
            int l = level(var(lits[i]));
            if (permDiff[l] != MYFLAG) {
          permDiff[l] = MYFLAG;
          nblevels++;
            }
          }
        } else { // -------- DEFAULT MODE. NOT A LOT OF DIFFERENCES... BUT EASIER TO READ
    #endif
        for(int i = 0; i < lits.size(); i++) {
            int l = level(var(lits[i]));
            if(permDiff[l] != MYFLAG) {
                permDiff[l] = MYFLAG;
                nblevels++;
            }
        }
    #ifdef INCREMENTAL
        }
    #endif
        return nblevels;
    }
};


//=================================================================================================
// Implementation of inline methods:

inline CRef Solver::reason(Var x) const { return vardata[x].reason; }

inline void Solver::insertVarOrder(Var x) {
    if (!order_heap.inHeap(x) && decision[x]) order_heap.insert(x); }

inline void Solver::varDecayActivity() { var_inc *= (1 / var_decay); }
inline void Solver::varBumpActivity(Var v) { varBumpActivity(v, var_inc); }
inline void Solver::varBumpActivity(Var v, double inc) {
    if ( (activity[v] += inc) > 1e100 ) {
        // Rescale:
        for (int i = 0; i < nVars(); i++)
            activity[i] *= 1e-100;
        var_inc *= 1e-100; }

    // Update order_heap with respect to new activity:
    if (order_heap.inHeap(v))
        order_heap.decrease(v); }

inline void Solver::claDecayActivity() { cla_inc *= (1 / clause_decay); }
inline void Solver::claBumpActivity (Clause& c) {
        if ( (c.activity() += cla_inc) > 1e20 ) {
            // Rescale:
            for (int i = 0; i < learned.size(); i++)
                ca[learned[i]].activity() *= 1e-20;
            cla_inc *= 1e-20; } }

inline void Solver::checkGarbage(void){ return checkGarbage(garbage_frac); }
inline void Solver::checkGarbage(double gf){
    if (ca.wasted() > ca.size() * gf)
        garbageCollect(); }

// NOTE: enqueue does not set the ok flag! (only public methods do)
inline bool     Solver::enqueue         (Lit p, CRef from)      { return value(p) != l_Undef ? value(p) != l_False : (uncheckedEnqueue(p, from), true); }
inline bool     Solver::addClause       (const vec<Lit>& ps)    { ps.copyTo(add_tmp); return addClause_(add_tmp); }
inline bool     Solver::addEmptyClause  ()                      { add_tmp.clear(); return addClause_(add_tmp); }
inline bool     Solver::addClause       (Lit p)                 { add_tmp.clear(); add_tmp.push(p); return addClause_(add_tmp); }
inline bool     Solver::addClause       (Lit p, Lit q)          { add_tmp.clear(); add_tmp.push(p); add_tmp.push(q); return addClause_(add_tmp); }
inline bool     Solver::addClause       (Lit p, Lit q, Lit r)   { add_tmp.clear(); add_tmp.push(p); add_tmp.push(q); add_tmp.push(r); return addClause_(add_tmp); }
 inline bool     Solver::locked          (const Clause& c) const { 
   if(c.size()>2) 
     return value(c[0]) == l_True && reason(var(c[0])) != CRef_Undef && ca.lea(reason(var(c[0]))) == &c; 
   return 
     (value(c[0]) == l_True && reason(var(c[0])) != CRef_Undef && ca.lea(reason(var(c[0]))) == &c)
     || 
     (value(c[1]) == l_True && reason(var(c[1])) != CRef_Undef && ca.lea(reason(var(c[1]))) == &c);
 }
inline void     Solver::newDecisionLevel()                      {
    if (decisionLevel() == 0) {
        stats[atLevel0] = trail.size();
    }
    trail_lim.push(trail.size());
}
inline int      Solver::level         (Var x) const   { return vardata[x].level; }
inline uint32_t Solver::abstractLevel (Var x) const   { return 1 << (level(x) & 31); }
inline lbool    Solver::value         (Var x) const   { return assigns[x]; }
inline lbool    Solver::value         (Lit p) const   { lbool res = assigns[var(p)] ^ sign(p);
    assert(res == l_True || res == l_False || res == l_Undef); return res; }
inline lbool    Solver::modelValue    (Var x) const   { return model[x]; }
inline lbool    Solver::modelValue    (Lit p) const   { return model[var(p)] ^ sign(p); }
inline int      Solver::nAssigns      ()      const   { return trail.size(); }
inline int      Solver::nClauses      ()      const   { return clauses.size(); }
inline int      Solver::nLearned      ()      const   { return learned.size(); }
inline int      Solver::totalClauseCount()    const   { return learned.size() + clauses.size() + permanentlyLearned.size(); }
inline int      Solver::nVars         ()      const   { return vardata.size(); }
inline int      Solver::nFreeVars     ()         { 
    uint64_t a = stats[decVars];
    return (int)(a) - (trail_lim.size() == 0 ? trail.size() : trail_lim[0]); }
inline void     Solver::setPolarity   (Var v, bool b) { polarity[v] = b; }
inline void     Solver::setDecisionVar(Var v, bool b) 
{ 
    if      ( b && !decision[v]) stats[decVars]++;
    else if (!b &&  decision[v]) stats[decVars]--;

    decision[v] = b;
    insertVarOrder(v);
}
inline void     Solver::setConfBudget(int64_t x){ conflict_budget    = conflicts    + x; }
inline void     Solver::setPropBudget(int64_t x){ propagation_budget = stats[propagations] + x; }
inline void     Solver::budgetOff(){ conflict_budget = propagation_budget = -1; }
inline bool     Solver::withinBudget() const {
    return 
           (conflict_budget    < 0 || conflicts < (uint64_t)conflict_budget) &&
           (propagation_budget < 0 || stats[propagations] < (uint64_t)propagation_budget); }

// FIXME: after the introduction of asynchronous interrruptions the solve-versions that return a
// pure bool do not give a safe interface. Either interrupts must be possible to turn off here, or
// all calls to solve must return an 'lbool'. I'm not yet sure which I prefer.
inline bool     Solver::solve         ()                    { budgetOff(); assumptions.clear(); return solve_() == l_True; }
inline bool     Solver::solve         (Lit p)               { budgetOff(); assumptions.clear(); assumptions.push(p); return solve_() == l_True; }
inline bool     Solver::solve         (Lit p, Lit q)        { budgetOff(); assumptions.clear(); assumptions.push(p); assumptions.push(q); return solve_() == l_True; }
inline bool     Solver::solve         (Lit p, Lit q, Lit r) { budgetOff(); assumptions.clear(); assumptions.push(p); assumptions.push(q); assumptions.push(r); return solve_() == l_True; }
inline bool     Solver::solve         (const vec<Lit>& assumps){ budgetOff(); assumps.copyTo(assumptions); return solve_() == l_True; }
inline lbool    Solver::solveLimited  (const vec<Lit>& assumps){ assumps.copyTo(assumptions); return solve_(); }
inline bool     Solver::okay          ()      const   { return ok; }

inline void     Solver::toDimacs     (const char* file){ vec<Lit> as; toDimacs(file, as); }
inline void     Solver::toDimacs     (const char* file, Lit p){ vec<Lit> as; as.push(p); toDimacs(file, as); }
inline void     Solver::toDimacs     (const char* file, Lit p, Lit q){ vec<Lit> as; as.push(p); as.push(q); toDimacs(file, as); }
inline void     Solver::toDimacs     (const char* file, Lit p, Lit q, Lit r){ vec<Lit> as; as.push(p); as.push(q); as.push(r); toDimacs(file, as); }



//=================================================================================================
// Debug etc:


inline void Solver::printLit(Lit l)
{
    printf("%s%d:%c", sign(l) ? "-" : "", var(l)+1, value(l) == l_True ? '1' : (value(l) == l_False ? '0' : 'X'));
}


inline void Solver::printClause(CRef cr)
{
  Clause &c = ca[cr];
    for (int i = 0; i < c.size(); i++){
        printLit(c[i]);
        printf(" ");
    }
}

inline void Solver::printInitialClause(CRef cr)
{
  Clause &c = ca[cr];
    for (int i = 0; i < c.size(); i++){
      if(!isSelector(var(c[i]))) {
    printLit(c[i]);
        printf(" ");
      }
    }
}

//=================================================================================================
struct reduceDBAct_lt {
    ClauseAllocator& ca;

    reduceDBAct_lt(ClauseAllocator& ca_) : ca(ca_) {
    }

    bool operator()(CRef x, CRef y) {

        // Main criteria... Like in MiniSat we keep all binary clauses
        if (ca[x].size() > 2 && ca[y].size() == 2) return 1;

        if (ca[y].size() > 2 && ca[x].size() == 2) return 0;
        if (ca[x].size() == 2 && ca[y].size() == 2) return 0;

        return ca[x].activity() < ca[y].activity();
    }
};

struct reduceDB_lt {
    ClauseAllocator& ca;

    reduceDB_lt(ClauseAllocator& ca_) : ca(ca_) {
    }

    bool operator()(CRef x, CRef y) {

        // Main criteria... Like in MiniSat we keep all binary clauses
        if (ca[x].size() > 2 && ca[y].size() == 2) return 1;

        if (ca[y].size() > 2 && ca[x].size() == 2) return 0;
        if (ca[x].size() == 2 && ca[y].size() == 2) return 0;

        // Second one  based on literal block distance
        if (ca[x].lbd() > ca[y].lbd()) return 1;
        if (ca[x].lbd() < ca[y].lbd()) return 0;


        // Finally we can use old activity or size, we choose the last one
        return ca[x].activity() < ca[y].activity();
        //return x->size() < y->size();

        //return ca[x].size() > 2 && (ca[y].size() == 2 || ca[x].activity() < ca[y].activity()); } 
    }


};



// The point for these three methods to have templates it that it can also work on MultiSolver which is not a solver
template<typename Solver> void printVarsClsCount(Solver &s) {
    if (s.verbosity() > 0) {
         printf("c |  Number of variables:  %12d                                                                   |\n", s.nVars());
         printf("c |  Number of clauses:    %12d                                                                   |\n", s.nClauses());
    }
}

template<typename Solver> void printParseTime(Solver &s, double time) {
    if (s.verbosity() > 0){
        printf("c |  Parse time:  %12.2f s                                                                          |\n", time);
    }
}

template<typename Solver> void printSimpTime(Solver &s, double time) {
    if (s.verbosity() > 0){
        printf("c |  Simplification time:  %12.2f s                                                                 |\n", time);
    }
}

template<typename Solver> void printProblemStatsHeader(Solver &s) {
    if (s.verbosity() > 0){
        printf("c ========================================[ Problem Statistics ]===========================================\n");
        printf("c |                                                                                                       |\n");
    }
}

double luby(double y, int x);

}


#endif
