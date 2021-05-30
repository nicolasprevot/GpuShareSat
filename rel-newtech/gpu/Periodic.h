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

#ifndef DEF_PERIODIC
#define DEF_PERIODIC

#include <vector>
#include <functional>

namespace Minisat {
class PeriodicChecker {
private:
    double startTime;
    // how many times we've already ran
    int runCount;
    int periodSec;

public:
    PeriodicChecker(int periodSec, double currentTime);
    // Returns 0 if we don't need to run, the nth time we ran otherwise
    int needToRun(double currentTime);
};

struct CheckerWithFunc {
    PeriodicChecker checker;
    std::function<void ()> func;
};

// This class is used to run some functions with specified periods, but only
// when someone calls maybeRun()
class PeriodicRunner {
private:
    double startTime;
    // I haven't been able to understand why, but it crashes if I use vec rather than std::vector
    std::vector<CheckerWithFunc> checkerWithFuncs;

public:
    PeriodicRunner(double _startTime);
    void add(int periodSec, std::function<void ()> func);
    void maybeRun(double currentTime);
};
}

#endif
