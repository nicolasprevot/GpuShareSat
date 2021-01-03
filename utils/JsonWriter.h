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

#ifndef DEF_JSON_WRITER
#define DEF_JSON_WRITER

#include <stdio.h>

namespace Glucose {

void writeJsonField(const char* name);

void setNeedNewlineAndComma();

// we can't use writeAsJson here because it wouldn't quote the value
void writeJsonString(const char *name, const char *val);

// here because Solver doesn't use printV
inline void writeAsJson(const char *name, unsigned long val) {
    writeJsonField(name);
    printf("%ld", val);
    setNeedNewlineAndComma();
}

inline void writeAsJson(const char *name, long val) {
    writeJsonField(name);
    printf("%ld", val);
    setNeedNewlineAndComma();
}

inline void writeAsJson(const char *name, double val) {
    writeJsonField(name);
    printf("%lf", val);
    setNeedNewlineAndComma();
}

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
}
#endif
