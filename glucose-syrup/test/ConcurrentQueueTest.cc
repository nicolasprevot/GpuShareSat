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

#include <boost/test/unit_test.hpp>
#include "../gpuShareLib/ConcurrentQueue.h"

namespace GpuShare {

BOOST_AUTO_TEST_SUITE( ConcurrentQueueTest )

BOOST_AUTO_TEST_CASE(maxAndMinTest) {
    ConcurrentQueue<int> cq(2);
    // add three elements
    cq.getNew() = 3;
    cq.addNew();
    cq.getNew() = 2;
    cq.addNew();
    cq.getNew() = 5;
    cq.addNew();

    // get and remove them
    int *pt = NULL;

    BOOST_CHECK(cq.getIncrInter(pt));
    BOOST_CHECK_EQUAL(3, *pt);
    BOOST_CHECK(cq.getIncrInter(pt));
    BOOST_CHECK_EQUAL(2, *pt);
    BOOST_CHECK(cq.getIncrInter(pt));
    BOOST_CHECK_EQUAL(5, *pt);
    BOOST_CHECK(!cq.getIncrInter(pt));

    BOOST_CHECK(cq.getMin(pt));
    BOOST_CHECK_EQUAL(3, *pt);
    cq.removeMin();
    BOOST_CHECK(cq.getMin(pt));
    BOOST_CHECK_EQUAL(2, *pt);
    cq.removeMin();
    BOOST_CHECK(cq.getMin(pt));
    BOOST_CHECK_EQUAL(5, *pt);
    cq.removeMin();
    BOOST_CHECK(!cq.getMin(pt));

    // add one more
    cq.getNew() = 9;
    cq.addNew();

    BOOST_CHECK(cq.getIncrInter(pt));
    BOOST_CHECK_EQUAL(9, *pt);

    BOOST_CHECK(cq.getMin(pt));
    BOOST_CHECK_EQUAL(9, *pt);
    cq.removeMin();
    BOOST_CHECK(!cq.getMin(pt));
}

BOOST_AUTO_TEST_SUITE_END()

}
