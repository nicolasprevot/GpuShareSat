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

// This purposedly doesn't have a header guard
// It's using X macros
// The point is to only write all the stats once
X(gpuClauses)
X(gpuClauseLengthSum)
X(gpuClausesAdded)
X(gpuRuns)
// When we test 32 groups at once, it only counts towards 1 here
X(clauseTestsOnGroups)
// When we test 32 assignments at once, it only counts towards 1 here
// Getting this stat is slow compared to the others
X(clauseTestsOnAssigs)
X(totalAssigClauseTested)
X(gpuReduceDbs)
X(gpuReports)
X(timeSpentTestingClauses)
X(timeSpentFillingAssigs)
X(timeSpentFillingReported)
X(timeSpentReduceGpuDb)
