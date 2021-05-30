/***************************************************************************************

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

#include "Periodic.h"

namespace Minisat {

PeriodicChecker::PeriodicChecker(int _periodSec, double _startTime):
    startTime(_startTime),
    runCount(0), 
    periodSec(_periodSec) {
}

int PeriodicChecker::needToRun(double currentTime) {
    if (periodSec < 0) return 0;
    if (currentTime >= startTime + (runCount + 1) * periodSec) {
        return ++runCount;
    }
    return 0;
}

PeriodicRunner::PeriodicRunner(double _startTime): startTime(_startTime) {
}

void PeriodicRunner::add(int periodSec, std::function<void ()> func) {
    checkerWithFuncs.push_back(CheckerWithFunc {PeriodicChecker(periodSec, startTime), func});
}

void PeriodicRunner::maybeRun(double currentTime) {
    for (unsigned int i = 0; i < checkerWithFuncs.size(); i++) {
        if (checkerWithFuncs[i].checker.needToRun(currentTime)) {
            checkerWithFuncs[i].func();
        }
    }
}

}
