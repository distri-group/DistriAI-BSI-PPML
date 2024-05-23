/*
 * Generator.h
 *
 */

#ifndef FHE_GENERATOR_H_
#define FHE_GENERATOR_H_

#include <vector>
using namespace std;

#include "Math/BigInt.h"
#include "Math/modp.h"

template <class T>
class Generator
{
 public:
    virtual ~Generator() {}
    virtual Generator* clone() const = 0;
    virtual void get(T& x) const = 0;
};

template <class T>
class Iterator : public Generator<T>
{
    const vector<T>& v;
    mutable size_t i;

public:
    Iterator(const vector<T>& v) : v(v), i(0) {}
    Generator<T>* clone() const { return new Iterator(*this); }
    void get(T& x) const { x = v[i]; i++; }
};

class ConversionIterator : public Generator<BigInt>
{
    const vector<modp>& v;
    const Zp_Data& ZpD;
    mutable size_t i;

public:
    ConversionIterator(const vector<modp>& v, const Zp_Data& ZpD) : v(v), ZpD(ZpD), i(0) {}
    Generator<BigInt>* clone() const { return new ConversionIterator(*this); }
    void get(BigInt& x) const { to_bigint(x, v[i], ZpD); i++; }
};

class WriteConversionIterator : public Generator<BigInt>
{
    vector<modp>& v;
    const Zp_Data& ZpD;
    mutable size_t i;
    mutable BigInt tmp;

public:
    WriteConversionIterator(vector<modp>& v, const Zp_Data& ZpD) : v(v), ZpD(ZpD), i(0) {}
    Generator<BigInt>* clone() const { return new WriteConversionIterator(*this); }
    void get(BigInt& x) const { tmp = x; v[i].convert_destroy(tmp, ZpD); i++; }
};

#endif /* FHE_GENERATOR_H_ */
