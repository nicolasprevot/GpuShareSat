/***************************************************************************************
 GpuShareSat -- Copyright (c) 2020, Nicolas Prevot

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

#include <boost/test/unit_test.hpp>
#include "../core/Solver.h"
#include "../core/Finisher.h"
#include "../core/Dimacs.h"

namespace Minisat {

BOOST_AUTO_TEST_SUITE( SolverTest )

BOOST_AUTO_TEST_CASE(SolverSolveTest) {
        Finisher finisher;
        GpuShare::GpuClauseSharer sharer;
        GpuShare::Logger logger {2, GpuShare::directPrint};
        Solver solver(10, sharer, finisher, false, logger);
        solver.newVar();
        solver.newVar();

        solver.addClause(~mkLit(0));
        solver.addClause(mkLit(1));

        lbool ret = solver.solve();
        BOOST_CHECK(ret == l_True);
        BOOST_CHECK((l_False == solver.modelValue(0)));
        BOOST_CHECK((l_True == solver.modelValue(1)));
}

// Propagation
BOOST_AUTO_TEST_CASE(SolverPropagate) {
        Finisher finisher;
        GpuShare::GpuClauseSharer sharer;
        GpuShare::Logger logger {2, GpuShare::directPrint};
        Solver solver(0, sharer, finisher, false, logger);
        solver.newVar();
        solver.newVar();

        solver.addClause(~mkLit(0), mkLit(1));
        solver.enqueue(mkLit(0));

        BOOST_CHECK((l_True == solver.value(0)));
        BOOST_CHECK((l_Undef == solver.value(1)));
        solver.propagate();
        BOOST_CHECK((l_True == solver.value(0)));
        BOOST_CHECK((l_True == solver.value(1)));

}
BOOST_AUTO_TEST_SUITE_END()
}
