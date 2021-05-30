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

#ifndef DEF_JSON_WRITER
#define DEF_JSON_WRITER

#include <stdio.h>
#include <fstream>
#include <memory>

namespace Glucose {

class JsonWriter;

class JObj {
private:
    JsonWriter &writer;
public:
    JObj(JsonWriter &writer);
    ~JObj();
};
// This is meant to be the outside object for a whole set of stats
class JStats {
private:
    JObj *jo;
    std::ostream &ost;
public:
    JStats(JsonWriter &writer, std::ostream &ost);
    ~JStats();
};

class JArr {
private:
    JsonWriter &writer;
public:
    JArr(JsonWriter &writer);
    ~JArr();
};

class JsonWriter {
    private:
    bool needComma;
    bool needNewline;
    std::ostream &os;

    void writePrecJson();
    void write(const char *val);

    // disallow copying
    JsonWriter&  operator = (JsonWriter &other) = delete;
    JsonWriter(JsonWriter& other) = delete;

    public:
    
    void setNeedNewlineAndComma(bool v = true);
    JsonWriter(std::ostream &os);
    void writeJsonString(const char *name, const char *val); 
    void writeJsonField(const char *name);

    template<typename T> void write(const char *name, T val) {
        writeJsonField(name);
        os << val;
        setNeedNewlineAndComma();
    }
    ~JsonWriter();
    friend class JObj;
    friend class JArr;
};

// write json to a file, inside an array
class JsonStatsWriter {
    private:
    std::ofstream ostream;

    public:
    JsonWriter writer;
    JsonStatsWriter(const char *fileName);

    private:
    JArr jarr;

};

#define writeVarJson(jsonWriter, v) {\
    jsonWriter.write(#v, v);\
}

}
#endif
