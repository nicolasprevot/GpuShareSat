
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

#ifndef DEF_FINISHER
#define DEF_FINISHER

#include <limits>

namespace Glucose {
/* This struct is accessed by several threads but does not use locks */
struct Finisher {
    // It is fine not to guard this by a lock because:
    // - When writing, even if several thread write to it, the result will be one of these threads, which is valid.
    // - When readin, it is fine as long as the reader joins all the threads which may have written
    int oneThreadIdWhoFoundAnAnswer; // -1 if no thread has found an aswer

    // This is used by signal handlers which cannot use locks
    // It's fine not to guard by a lock because the initial value is false, and then it can only be set to true
    // After setting to true, it is possible that threads won't read true immediately, but it's fine
    volatile bool stopAllThreads;

    // It's fine not to guard by a lock because there's only one thread which changes it
    // After the value changes, it's possible that threads won't read the updated value immediately, but it's fine
    volatile int stopAllThreadsAfterId;

    Finisher(): oneThreadIdWhoFoundAnAnswer(-1), stopAllThreads(false), stopAllThreadsAfterId(std::numeric_limits<int>::max()) {
    }

    bool shouldIStop(int threadId) { return stopAllThreads || threadId >= stopAllThreadsAfterId; }
};
}

#endif
