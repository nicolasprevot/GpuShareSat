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

// This file is aiming at making assertions easier. It intended for cu or cuh files. It differs from AssertC.cuh in that there's no cout on the GPU so instead
// we use a printC function

#ifndef DEF_ASSERT_C
#define DEF_ASSERT_C

#include <assert.h>
#include "Assert.h"

#ifndef NDEBUG

#define PRINT_VALUES_MSG_C(var1, var2, msgExpr)\
    printf(" values are ");\
    PRINTCN(var1);\
    printf(" and ");\
    PRINTCN(var2);\
    printf(" ");\
    msgExpr;\
    printf("\n")

#define ASSERT_OP_MSG_C(var1, op, var2, msgExpr)\
    ASSERT_MSG((var1) op (var2), PRINT_VALUES_MSG_C(var1, var2, msgExpr));

#define ASSERT_OP_C(var1, op, var2)\
    ASSERT_OP_MSG_C(var1, op, var2, );

// inclusive for min, exclusive for max
#define ASSERT_BETWEEN_C(val, min, max)\
    ASSERT_OP_C(val, >=,  min);\
    ASSERT_OP_C(val, <, max);

#else

#define ASSERT_MSG_C(expr, msg)

#define ASSERT_OP_C(var1, op, var2)
#define ASSERT_OP_MSG_C(var1, op, var2, msg)

#define ASSERT_BETWEEN_C(val, min, max)

#endif

#endif
