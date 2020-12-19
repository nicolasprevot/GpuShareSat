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

#ifndef DEF_UTILS
#define DEF_UTILS
#include <time.h>
#include <mutex>
#include "Assert.h"
#include <functional>
#include <atomic>
#include <memory>

#define MAXIMUM_SLEEP_DURATION 5

namespace Glucose {

/* How printing works:
Requirements:
- We can do things like ASSERT_EQUAL(a, b) which will print a relevant error message with the values of a and b, whatever the type of a and b
- It can work on the host code on .cc files, on the gpu in .cu files, on the host in .cu files 
Decision:
- For all the types we care about, there is a printV(type) method defined only on the host, and a printVD(type) method defined only on the device
- we can use PRINTV() on host or on device, and it will do the right thing
*/

#define NL printf("\n");

#ifdef __CUDA_ARCH__
#define PRINTV printVD
#else
#define PRINTV printV
#endif

#define PRINT(x)\
printf(#x ": ");\
PRINTV(x);\
printf(" ");

#define PRINTLN(x) { PRINT(x); NL }

timespec getNextTimePrintStats();

bool operator ==(const timespec& lhs, const timespec& rhs);
bool operator !=(const timespec& lhs, const timespec& rhs);

// The point of this class is to allow synchronized writing to stdout
class SyncOut {
private:
    static std::mutex lock;
    std::lock_guard<std::mutex> lockGuard;
public:
    SyncOut(): lockGuard(lock) {
    }
};

#define SYNCED_OUT(toRun) { SyncOut so; toRun;}

inline void setOnMaskUint(uint &val, uint mask, uint cond) {
    if (cond) {
        val = val | mask;
    }
    else {
        val = val & ~mask;
    }
}

int randBetween(int min, int max);

void printV(long v);
void printV(unsigned long v);
void printV(int v);
void printV(uint v);
void printV(void* pt);
void printV(const char* chs);
void printV(float f);
void printV(double d);

inline bool hasNoMoreThanOneBit(uint x) {
    return (x & (x - 1)) == 0;
}

inline int countBitsSet(uint x) {
    int res = 0;
    while (x != 0) {
        res++;
        x = x & (x - 1);
    }
    return res;
}

inline void assertHasExactlyOneBit(uint x) { 
    ASSERT_MSG(x != 0, PRINT(x));
    ASSERT_MSG(hasNoMoreThanOneBit(x), PRINT(x));
}

inline uint getFirstBit(uint x) {
    assert(x != 0);
    return x & ~(x - 1);
}

template<typename T> T mustSucceed(std::function<bool (T &val)> func) {
    T v;
    if (!func(v)) {
        THROW();
    }   
    return v;
}

#define MUST_SUCCEED(type, func, ...) \
mustSucceed(std::function<bool (type &)> ([&] (type &u) { return func(u, ##__VA_ARGS__); }))

void writeJsonField(const char* name);

void setNeedNewlineAndComma();

// we can't use writeAsJson here because it wouldn't quote the value
void writeJsonString(const char *name, const char *val);

template<typename T> void writeAsJson(const char *name, T val) {
    writeJsonField(name);
    printV(val);
    setNeedNewlineAndComma();
}

#define writeVarJson(v) {\
    writeAsJson(#v, v);\
}

class JObj {
public:
    JObj();
    ~JObj();
};

// This is meant to be the outside object for a whole set of stats
class JStats {
private:
    JObj *jo;
public:
    JStats();
    ~JStats();
};

class JArr {
public:
    JArr();
    ~JArr();
};

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

    Finisher();

    bool shouldIStop(int threadId) { return stopAllThreads || threadId >= stopAllThreadsAfterId; }
};

class TimePrinter {
private:
    double cpuTimeSecStarted;
    double realTimeSecStarted;
    const char *message;
public:
    TimePrinter(const char *message);
    ~TimePrinter();
};

inline bool bitsCommon(uint v1, uint v2) {
    return (v1 & v2) != 0;
}

// inclusive for start, not for end
inline uint bitsBetween(int start, int end) {
    // 1 << 32 overflows so we need to special case this
    if (start == 0 && end == sizeof(uint) * 8) return ~0;
    return ((1 << (end - start)) - 1) << start;
}

inline uint bitsUntil(int p) {
    // 1 << 32 overflows so we need to special case this
    if (p == sizeof(uint) * 8) return ~0;
    return (1 << p) - 1;
}

inline void makeSameAsF(uint &val, uint toSet, uint from) {
    assertHasExactlyOneBit(from);
    if ((val & from) != 0) {
        val = val | toSet;
    }
    else {
        val = val & ~toSet;
    }
}

void printBinary(uint x);
double luby(double y, int x);

// Returns a random float 0 <= x < 1. Seed must never be 0.
static inline double drand(double& seed) {
    seed *= 1389796;
    int q = (int)(seed / 2147483647);
    seed -= (double)q * 2147483647;
    return seed / 2147483647; }

// Returns a random integer 0 <= x < size. Seed must never be 0.
static inline int irand(double& seed, int size) {
    return (int)(drand(seed) * size);
}

// returns an integer min <= x < max. Seed must never be 0
static inline int irand(double& seed, int min, int max) {
    return irand(seed, max - min) + min;
}

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}


#endif
