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

#include "Profiler.h"
#include "Utils.h"
#include "System.h"

using namespace Glucose;

void Profiler::bump(std::string name, double time) {
    if (timesTaken.find(name) == timesTaken.end()) {
        timesTaken[name] = 0.0;
    }
    timesTaken[name] += time;
}

void Profiler::printStats() {
    for (auto &it : timesTaken) {
        writeAsJson(it.first.c_str(), it.second);
    }
}


TimeGauge::TimeGauge(Profiler &_profiler, std::string _name, bool enabled): 
    profiler(_profiler), name(_name), timeStarted(enabled ? realTimeSec() : -1) {
}

void TimeGauge::complete() {
    if (timeStarted >= 0) profiler.bump(name, realTimeSec() - timeStarted);
    timeStarted = -1;
}

TimeGauge::~TimeGauge() {
    complete();
}
