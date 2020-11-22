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

#ifndef DEF_ASSERT
#define DEF_ASSERT
#include <assert.h>

#define THROW_ERROR(msgExpr) {\
    printf("Error in %s:%d: ", __FILE__, __LINE__);\
    msgExpr;\
    printf("\n");\
    THROW();\
}

#ifdef __CUDA_ARCH__
// can't use exit (host function) from gpu. This will just do nothing in release
#define THROW() assert(false)
#else
// If ndebug isn't set, we can use assert(false), it's better when debugging
// because it will stop execution there. Otherwise, we need exit
#ifdef NDEBUG 
#define THROW() exit(1)
#else 
#define THROW() assert(false)
#endif
#endif

#ifndef NDEBUG

#define ASSERT_OR_DO(expr) assert(expr);

#define ASSERT(expr)\
    if (!(expr)) THROW_ERROR(printf("assertion failed: " #expr "\n"));

#define ASSERT_MSG(expr, msgExpr)\
    if (!(expr)) THROW_ERROR(printf("assertion failed: " #expr " message: "); msgExpr);

#define PRINT_VALUES_MSG(var1, var2, msgExpr)\
    printf(" values are ");\
    PRINTV(var1);\
    printf(" and ");\
    PRINTV(var2);\
    printf(" ");\
    msgExpr;\
    printf("\n")

#define ASSERT_OP_MSG(var1, op, var2, msgExpr)\
    ASSERT_MSG((var1) op (var2), PRINT_VALUES_MSG(var1, var2, msgExpr));

#define ASSERT_OP(var1, op, var2)\
    ASSERT_OP_MSG(var1, op, var2, );

// inclusive for min, exclusive for max
#define ASSERT_BETWEEN(val, min, max)\
    ASSERT_OP(val, >=,  min);\
    ASSERT_OP(val, <, max);

#else

#define ASSERT_OR_DO(expr) expr;

#define ASSERT(expr)
#define ASSERT_MSG(expr, msg)

#define ASSERT_OP(var1, op, var2)
#define ASSERT_OP_MSG(var1, op, var2, msg)

#define ASSERT_BETWEEN(val, min, max)

#endif

#endif
