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

#include "JsonWriter.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

namespace Minisat {

JsonWriter::JsonWriter(std::ostream &_os): needComma(false), needNewline(false), os(_os) {
}

void JsonWriter::setNeedNewlineAndComma() {
    needComma = true;
    needNewline = true;
}

void JsonWriter::writeJsonString(const char *name, const char *val) {
    writeJsonField(name);
    os << "\"" << val << "\"";
    setNeedNewlineAndComma();
}

void JsonWriter::write(const char *val) {
    os << val;
}

void JsonWriter::writePrecJson() {
    if (needComma) {
        os << ",";
    }
    if (needNewline) {
        os << "\n";
    }
    needComma = false;
    needNewline = false;
}

void JsonWriter::writeJsonField(const char* name) {
    writePrecJson();
    os << "\"" << name << "\"" << ":";
}

JsonWriter::~JsonWriter() {
    if (needNewline) {
        write("\n");
    }
}

JObj::JObj(JsonWriter &_writer): writer(_writer) {
    writer.writePrecJson();
    writer.write("{");
    writer.needNewline = true;
}

JObj::~JObj() {
    if (writer.needNewline) {
        writer.write("\n");
    }
    writer.write("}");
    writer.setNeedNewlineAndComma();
}

JArr::JArr(JsonWriter &_writer): writer(_writer) {
    writer.writePrecJson();
    writer.write("[");
    writer.needNewline = true;
}

JArr::~JArr() {
    if (writer.needNewline) {
        writer.write("\n");
    }
    writer.write("]");
    writer.setNeedNewlineAndComma();
}

JsonStatsWriter::JsonStatsWriter(const char *fileName): 
    ostream(fileName, std::ofstream::out), 
    writer(ostream), 
    jarr(writer) {
}

}
