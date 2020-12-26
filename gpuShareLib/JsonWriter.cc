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

#include "JsonWriter.h"
#include <stdio.h>

namespace GpuShare {

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

}
