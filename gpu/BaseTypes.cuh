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

#ifndef DEF_BASETYPES_CUH
#define DEF_BASETYPES_CUH
#include "Helper.cuh"
#include "satUtils/SolverTypes.h"

namespace Glucose {

#define MAX_CL_SIZE 100

typedef uint64_t GpuClauseId;

typedef uint32_t Vals;

typedef unsigned int ValsACas;


struct GpuCref {
    int clSize;
    int clIdInSize;
};

struct AssigIdsPerSolver {
    int startAssigId;
    int assigCount;

    int getId(int position) {
        int s = sizeof(Vals) * 8;
        int startAssigPos = startAssigId % s;
        if (startAssigPos <= position) {
            return startAssigId + position - startAssigPos;
        }
        return startAssigId + position - startAssigPos + s;
    }
};


inline bool operator==(const GpuCref cref1, const GpuCref cref2) {
    return cref1.clSize == cref2.clSize && cref1.clIdInSize == cref2.clIdInSize;
}

struct MultiLBool {
    Vals isDef;
    // if isDef is false for a bit, then the value of isTrue can be anything
    Vals isTrue;

    bool operator==(const MultiLBool& other) const {
        return isDef == other.isDef && ((isTrue & isDef) == (other.isTrue & other.isDef));
    }

    Vals withFalse() const { return isDef & ~isTrue; }

    Vals withTrue() const { return isDef & isTrue; }

    Vals withUndef() const { return ~isDef; }

    lbool getLBool(Vals mask) {
        assertHasExactlyOneBit(mask);
        if ((isDef & mask) == 0) return l_Undef;
        if ((isTrue & mask) == 0) return l_False;
        return l_True;
    }

    void printBinary() {
        printf("isDef: "); Glucose::printBinary(isDef); NL;
        printf("isTrue: "); Glucose::printBinary(isTrue); NL;
    }

    lbool getUniqueVal() {
        if (isDef == 0u) return l_Undef;
        if (isDef != ~0u) return l_Inexisting;
        if (isTrue == 0u) {
            return l_False;
        }
        if (isTrue != ~0U) return l_Inexisting;
        return l_True;
    }
};

inline MultiLBool operator~(MultiLBool vad) {
    return MultiLBool { vad.isDef, ~vad.isTrue };
}

inline MultiLBool operator<<(MultiLBool vad, int p) {
    return MultiLBool { vad.isDef << p, vad.isTrue << p };
}

inline MultiLBool operator>>(MultiLBool vad, int p) {
    return MultiLBool { vad.isDef >> p, vad.isTrue >> p };
}

inline MultiLBool operator&(MultiLBool vad, Vals u) {
    return MultiLBool { vad.isDef & u, vad.isTrue & u};
}

inline MultiLBool makeMultiLBool(lbool lb) {
    return MultiLBool { lb != l_Undef ? ~ 0u : 0u, lb == l_True ? ~0u : 0u};
}

inline void printV(const MultiLBool vad) {
	Vals tr = vad.withTrue();
	if (tr != 0) {
		printf("tr: "); Glucose::printBinary(tr); printf(" ");
	}
	Vals fa = vad.withFalse();
	if (fa != 0) {
		printf("fa: "); Glucose::printBinary(fa); printf(" ");
	}
}

__device__ inline Vals getTrue(MultiLBool vad) {
    return vad.isTrue & vad.isDef;
}

__device__ inline Vals getFalse(MultiLBool vad) {
    return ~vad.isTrue & vad.isDef;
}

// The reasons to pass the wrong assignments as an int and find the individual instances on
// the cpu rather than the gpu are:
// - it avoids computing them on the gpu which blocks 31 other threads of the wrap
// - easier to copy to the cpu, less to copy
struct ReportedClause {
    // an int, where every bit represents if the clause was wrong for each assignment
    Vals reportedAssignments;
    int solverId;
    GpuCref gpuCref;
};

inline __device__ __host__ bool dSign(Lit p){
    return p.x & 1;
}

inline __device__ __host__ uint dVar(Lit p) {
    return p.x >> 1;
}

inline __device__ __host__ bool isDefined(lbool lb) {
    return !(lb.value & 2);
}

inline __host__ __device__ bool value(lbool lb) {
    return !(lb.value & 1);
}

inline __host__ __device__ uint8_t toUint8(lbool lb) {
    return lb.value;
}

inline __host__ __device__ lbool mklbool(uint8_t v) {
    // This is a bit horrible, but lbool already has a constructors that only work on the host
    // so I need this
    return *(reinterpret_cast<lbool*>(&v));
}

__device__ void printVD(Lit l);

}

#endif
