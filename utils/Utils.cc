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

#include "utils/Utils.h"
#include "utils/ParseUtils.h"
#include <random>
#include <assert.h>
#include "utils/System.h"

namespace Glucose {

std::mutex SyncOut::lock;

timespec getNextTimePrintStats() {
    struct timespec timeout;
    time(&timeout.tv_sec);
    timeout.tv_sec += MAXIMUM_SLEEP_DURATION;
    timeout.tv_nsec = 0;
    return timeout;
}


Glucose::StreamBuffer::StreamBuffer(gzFile i) : in(i), pos(0), size(0) {
     if (pos >= size) {
         pos  = 0;
         size = gzread(in, buf, sizeof(buf));
     }
}

bool operator ==(const timespec& lhs, const timespec& rhs)
{
    return lhs.tv_sec == rhs.tv_sec && lhs.tv_nsec == rhs.tv_nsec;
}

bool operator !=(const timespec& lhs, const timespec& rhs)
{
    return !(lhs == rhs);
}

int randBetween(int min, int max) {
    return rand() % (max - min) + min;
}

void printV(long v) {
    printf("%ld ", v);
}

void printV(unsigned long v) {
    printf("%ld ", v);
}

void printV(void* pt) {
    printf("%p ", pt);
}

void printV(int v) {
    printf("%d ", v);
}

void printV(uint v) {
    printf("%x ", v);
}

void printV(const char* chs) {
    printf("%s", chs);
}

void printV(float f) {
    printf("%f", f);
}

void printV(double d) {
    printf("%lf", d);
}

// global variables
bool needCommaJson = false;
bool needNewlineJson = false;

void setNeedNewlineAndComma() {
    needCommaJson = true;
    needNewlineJson = true;
}

void writeJsonString(const char *name, const char *val) {
    writeJsonField(name);
    printf("\"%s\"", val);
    setNeedNewlineAndComma();
}

void writePrecJson() {
    if (needCommaJson) {
        printf(",");
    }
    if (needNewlineJson) {
        printf("\nc ");
    }
    needCommaJson = false;
    needNewlineJson = false;
}

void writeJsonField(const char* name) {
    writePrecJson();
    printf("\"%s\": ", name);
}

JStats::JStats() {
    printf("c stats_start\nc");
    needCommaJson = false;
    needNewlineJson = false;
    jo = new JObj();
}

JStats::~JStats() {
    delete jo;
    printf("\nc stats_end\n");
}

JObj::JObj() {
    writePrecJson();
    printf("{");
    needNewlineJson = true;
}

JObj::~JObj() {
    if (needNewlineJson) {
        printf("\nc ");
    }
    printf("}");
    needNewlineJson = true;
    needCommaJson = true;
}

JArr::JArr() {
    writePrecJson();
    printf("[");
    needNewlineJson = true;
}

JArr::~JArr() {
    if (needNewlineJson) {
        printf("\nc ");
    }
    printf("]");
    needNewlineJson = true;
    needCommaJson = true;
}

Finisher::Finisher() {
    finished = false;
    oneWhoHasFinished = -1;
    canceled = false;
    assert(canceled.is_lock_free());
}

void Finisher::iveFinished(int id) {
    {
        SyncOut so;
        printf("c thread %d has finished\n", id);
    }
    std::lock_guard<std::mutex> lockGuard(lock);
    if (!finished && !canceled) {
        finished = true;
        oneWhoHasFinished = id;
    }
}

int Finisher::getOneWhoHasFinished() {
    std::lock_guard<std::mutex> lockGuard(lock);
    assert(finished);
    assert(!canceled.load());
    assert(oneWhoHasFinished >= 0);
    return oneWhoHasFinished;
}

void Finisher::cancel() {
    canceled.store(true);
}

bool Finisher::hasCanceledOrFinished()  {
    return canceled.load() || finished;
}

bool Finisher::isCanceled() {
    return canceled.load();
}

TimePrinter::TimePrinter(const char *_message) : message(_message) {
    cpuTimeSecStarted = cpuTimeSec();
    realTimeSecStarted = realTimeSec();
}

TimePrinter::~TimePrinter() {
    printf("c cpu time %s: %f s\n", message, cpuTimeSec() - cpuTimeSecStarted);
    printf("c real time %s: %f s\n", message, realTimeSec() - realTimeSecStarted);
}

void printBinary(uint x) {
    bool seen1 = false;
    for (int i = 31; i >= 0; i--) {
        if ((1 << i) & x) {
            printf("1");
            seen1 = true;
        }
        else if (seen1) printf("0");
    }
}

double luby(double y, int x) {
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


}
