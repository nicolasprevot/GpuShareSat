/*******************************************************************************************[Vec.h]
Copyright (c) 2003-2007, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

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

#ifndef GpuShare_Vec_h
#define GpuShare_Vec_h

#include <assert.h>
#include <new>

#include "mtl/IntTypes.h"
#include "mtl/XAlloc.h"
#include <initializer_list>


#include <string.h>

namespace GpuShare {

//=================================================================================================
// Automatically resizable arrays
//
// NOTE! Don't use this vector on datatypes that cannot be re-located in memory (with realloc)

template<class T>
class vec {
    T*  data;
    int sz;
    int cap;

    // Don't allow copying (error prone):
    vec<T>&  operator = (vec<T>& other) { assert(0); return *this; }
             vec        (vec<T>& other) { assert(0); }
             
    // Helpers for calculating next capacity:
    static inline int  imax   (int x, int y) { int mask = (y-x) >> (sizeof(int)*8-1); return (x&mask) + (y&(~mask)); }
    //static inline void nextCap(int& cap){ cap += ((cap >> 1) + 2) & ~1; }
    static inline void nextCap(int& cap){ cap += ((cap >> 1) + 2) & ~1; }

public:
    // Constructors:
    vec()                       : data(NULL) , sz(0)   , cap(0)    { }
    explicit vec(int size)      : data(NULL) , sz(0)   , cap(0)    { growTo(size); }
    vec(int size, const T& pad) : data(NULL) , sz(0)   , cap(0)    { growTo(size, pad); }
   ~vec()                                                          { clear(true); }

    vec(std::initializer_list<T> l): vec() {
        for (auto ptr = l.begin(); ptr < l.end(); ptr++) {
            push(*ptr);
        }
    }

    T* getData() { return data; }

    bool compareWith(const vec<T> &other) {
        if (size() != other.size()) return false;
        return memcmp(data, other.data, size() * sizeof(T)) == 0;
    }

    // Size operations:
    int      size     (void) const     { return sz; }
    // reduce size by nelems
    void     shrink   (int nelems)     { assert(nelems <= sz); for (int i = 0; i < nelems; i++) sz--, data[sz].~T(); }
    void     shrink_  (int nelems)     { assert(nelems <= sz); sz -= nelems; }
    int      capacity (void) const     { return cap; }
    void     capacity (int min_cap);
    void     growTo   (int size);
    void     growTo   (int size, const T& pad);
    void     resize   (int newSize) { if (newSize < sz) shrink(sz - newSize); if (newSize > sz) growTo(newSize); }
    template<class ...Args> void growToInit(int size, Args... args);
    void growToNoInit(int size);
    void     clear    (bool dealloc = false);

    // Stack interface:
    void     push  (void)              { if (sz == cap) capacity(sz+1); new (&data[sz]) T(); sz++; }
    void     push  (const T& elem)     { if (sz == cap) capacity(sz+1); data[sz++] = elem; }
    void     push_ (const T& elem)     { assert(sz < cap); data[sz++] = elem; }
    void     pop   (void)              { assert(sz > 0); sz--, data[sz].~T(); }

    bool contains  (const T& elem) const {
        for (int i = 0; i < size(); i++) {
            if (operator[](i) == elem) return true; 
        }
        return false;
    }
    
    // returns if the element was there and has been removed
    bool     remove(const T &elem) {
        int tmp;
        for(tmp = 0;tmp<sz;tmp++) {
            if(data[tmp]==elem) 
                break;
        }
        if(tmp<sz) {
            assert(data[tmp] == elem);
            data[tmp] = data[sz-1];
            sz = sz - 1;
            return true;
        }
        return false;
    }
    
    // NOTE: it seems possible that overflow can happen in the 'sz+1' expression of 'push()', but
    // in fact it can not since it requires that 'cap' is equal to INT_MAX. This in turn can not
    // happen given the way capacities are calculated (below). Essentially, all capacities are
    // even, but INT_MAX is odd.

    const T& last  (void) const        { return data[sz-1]; }
    T&       last  (void)              { return data[sz-1]; }

    // Vector interface:
    const T& operator [] (int index) const { assert(index >= 0); assert(index < sz); return data[index]; }
    T&       operator [] (int index)       { assert(index >= 0); assert(index < sz); return data[index]; }

    // Duplicatation (preferred instead):
    void copyTo(vec<T>& copy) const { copy.clear(); copy.growTo(sz); for (int i = 0; i < sz; i++) copy[i] = data[i]; }

    void moveTo(vec<T>& dest) { dest.clear(true); dest.data = data; dest.sz = sz; dest.cap = cap; data = NULL; sz = 0; cap = 0; }
    void memCopyTo(vec<T>& copy) const{
        copy.capacity(cap);
        copy.sz = sz;
        memcpy(copy.data,data,sizeof(T)*cap);
    }

    bool isEqual(const vec<T> &other);
};

template<typename T> void clearObj(vec<T> &v) {
    v.clear(false);
}

template<class T> void printV(const vec<T>& vec) {
    printf("[");
    for (int i = 0; i < vec.size(); i++) {
        printV(vec[i]);
        if (i != vec.size() - 1) {
            printf(", ");
        }
    }
    printf("]");
}

class OutOfMemoryException {};

template<class T>
void vec<T>::capacity(int min_cap) {
    if (cap >= min_cap) return;
    int add = imax((min_cap - cap + 1) & ~1, ((cap >> 1) + 2) & ~1);   // NOTE: grow by approximately 3/2
    // occlists from simpSolver can be vec of vec and vec isn't classified as trivially copiable, but it's actually
    // fine to just realloc a vec
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wclass-memaccess"
    if (add > INT_MAX - cap || ((data = (T*)::realloc(data, (cap += add) * sizeof(T))) == NULL) && errno == ENOMEM)
        throw OutOfMemoryException();
    #pragma GCC diagnostic pop
 }


template<class T>
void vec<T>::growTo(int size, const T& pad) {
    if (sz >= size) return;
    capacity(size);
    for (int i = sz; i < size; i++) data[i] = pad;
    sz = size; }

template<class T> template<class... Args>
void vec<T>::growToInit(int size, Args... args) {
    if (sz >= size) return;
    capacity(size);
    for (int i = sz; i < size; i++) new (&data[i]) T(args...);
    sz = size; }

template<class T>
void vec<T>::growToNoInit(int size) {
    if (sz >= size) return;
    capacity(size);
    sz = size; }

template<class T>
void vec<T>::growTo(int size) {
    if (sz >= size) return;
    capacity(size);
    for (int i = sz; i < size; i++) new (&data[i]) T();
    sz = size; }

template<class T>
void vec<T>::clear(bool dealloc) {
    if (data != NULL){
        for (int i = 0; i < sz; i++) data[i].~T();
        sz = 0;
        if (dealloc) free(data), data = NULL, cap = 0; } }

//=================================================================================================
}

#endif