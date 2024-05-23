#ifndef _bigint
#define _bigint

#include <iostream>
using namespace std;

#include <stddef.h>
#include <gmpxx.h>

#include "Tools/Exceptions.h"
#include "Tools/int.h"
#include "Tools/random.h"
#include "Tools/octetStream.h"
#include "Tools/avx_memcpy.h"
#include "Protocols/config.h"

enum ReportType
{
  CAPACITY,
  USED,
  MINIMAL,
  REPORT_TYPE_MAX
};

template<int X, int L>
class gfp_;
template<int X, int L>
class gfpvar_;
class gmp_random;
class Integer;
template<int K> class Z2;
template<int K> class SignedZ2;
template<int L> class fixint;

namespace GC
{
  class Clear;
}

/**
 * Type for arbitrarily large integers.
 * This is a sub-class of ``thm_class`` from GMP. As such, it implements
 * all integers operations and input/output via C++ streams. In addition,
 * the ``get_ui()`` member function allows retrieving the least significant
 * 64 bits.
 */
class BigInt : public thm_class
{
public:
  static thread_local BigInt tmp, tmp2;
  static thread_local gmp_random random;

  // workaround for GCC not always initializing thread_local variables
  static void init_thread() { tmp = 0; tmp2 = 0; }

  template<class T>
  static mpf_class get_float(T v, T p, T z, T s);
  template<class U, class T>
  static void output_float(U& o, const mpf_class& x, T nan);

  /// Initialize to zero.
  BigInt() : thm_class() {}
  template <class T>
  BigInt(const T& x) : thm_class(x) {}
  /// Convert to canonical representation as non-negative number.
  template<int X, int L>
  BigInt(const gfp_<X, L>& x);
  /// Convert to canonical representation as non-negative number.
  template<int X, int L>
  BigInt(const gfpvar_<X, L>& x);
  /// Convert to canonical representation as non-negative number.
  template <int K>
  BigInt(const Z2<K>& x);
  /// Convert to canonical representation as non-negative number.
  template <int K>
  BigInt(const SignedZ2<K>& x);
  template <int L>
  BigInt(const fixint<L>& x) : BigInt(typename fixint<L>::super(x)) {}
  BigInt(const Integer& x);
  BigInt(const GC::Clear& x);
  BigInt(const mp_limb_t* data, size_t n_limbs);

  BigInt& operator=(int n);
  BigInt& operator=(long n);
  BigInt& operator=(word n);
  BigInt& operator=(double f);
  template<int X, int L>
  BigInt& operator=(const gfp_<X, L>& other);
  template<int K>
  BigInt& operator=(const Z2<K>& x);
  template<int K>
  BigInt& operator=(const SignedZ2<K>& x);

  /// Convert to signed representation in :math:`[-p/2,p/2]`.
  template<int X, int L>
  BigInt& from_signed(const gfp_<X, L>& other);
  template<class T>
  BigInt& from_signed(const T& other);

  void allocate_slots(const BigInt& x) { *this = x; }
  int get_min_alloc() { return get_thm_t()->_mp_alloc; }

  void mul(const BigInt& x, const BigInt& y) { *this = x * y; }

#ifdef REALLOC_POLICE
  ~BigInt() { lottery(); }
  void lottery();

  BigInt& operator-=(const BigInt& y)
  {
    if (rand() % 10000 == 0)
      if (get_thm_t()->_mp_alloc < abs(y.get_thm_t()->_mp_size) + 1)
        throw runtime_error("insufficient allocation");
    ((thm_class&)*this) -= y;
    return *this;
  }
  BigInt& operator+=(const BigInt& y)
  {
    if (rand() % 10000 == 0)
      if (get_thm_t()->_mp_alloc < abs(y.get_thm_t()->_mp_size) + 1)
        throw runtime_error("insufficient allocation");
    ((thm_class&)*this) += y;
    return *this;
  }
#endif

  int numBits() const
  { return thm_sizeinbase(get_thm_t(), 2); }

  void generateUniform(PRNG& G, int n_bits, bool positive = false)
  { G.get(*this, n_bits, positive); }

  void pack(octetStream& os, int = -1) const { os.store(*this); }
  void unpack(octetStream& os, int = -1)     { os.get(*this); };

  size_t report_size(ReportType type) const;
};


void inline_mpn_zero(mp_limb_t* x, mp_size_t size);
void inline_mpn_copyi(mp_limb_t* dest, const mp_limb_t* src, mp_size_t size);


inline BigInt& BigInt::operator=(int n)
{
  thm_class::operator=(n);
  return *this;
}

inline BigInt& BigInt::operator=(long n)
{
  thm_class::operator=(n);
  return *this;
}

inline BigInt& BigInt::operator=(word n)
{
  thm_class::operator=(n);
  return *this;
}

inline BigInt& BigInt::operator=(double f)
{
  thm_class::operator=(f);
  return *this;
}

template<int K>
BigInt::BigInt(const Z2<K>& x)
{
  *this = x;
}

template<int K>
BigInt& BigInt::operator=(const Z2<K>& x)
{
  thm_import(get_thm_t(), Z2<K>::N_WORDS, -1, sizeof(mp_limb_t), 0, 0, x.get_ptr());
  return *this;
}

template<int K>
BigInt::BigInt(const SignedZ2<K>& x)
{
  *this = x;
}

template<int K>
BigInt& BigInt::operator=(const SignedZ2<K>& x)
{
  thm_import(get_thm_t(), Z2<K>::N_WORDS, -1, sizeof(mp_limb_t), 0, 0, x.get_ptr());
  if (x.negative())
  {
    BigInt::tmp2 = 1;
    BigInt::tmp2 <<= K;
    *this -= BigInt::tmp2;
  }
  return *this;
}

template<int X, int L>
BigInt::BigInt(const gfp_<X, L>& x)
{
  *this = x;
}

template<int X, int L>
BigInt::BigInt(const gfpvar_<X, L>& other)
{
  to_bigint(*this, other.get(), other.get_ZpD());
}

template<int X, int L>
BigInt& BigInt::operator=(const gfp_<X, L>& x)
{
  to_bigint(*this, x);
  return *this;
}

template<class T>
void to_bigint(BigInt& res, const T& other)
{
  other.to(res);
}

template<class T>
void to_gfp(T& res, const BigInt& a)
{
  res = a;
}

string to_string(const BigInt& x);

/**********************************
 *       Utility Functions        *
 **********************************/

inline int gcd(const int x,const int y)
{
  BigInt& xx = BigInt::tmp = x;
  return thm_gcd_ui(NULL,xx.get_thm_t(),y);
}


inline BigInt gcd(const BigInt& x,const BigInt& y)
{ 
  BigInt g;
  thm_gcd(g.get_thm_t(),x.get_thm_t(),y.get_thm_t());
  return g;
}


inline void invMod(BigInt& ans,const BigInt& x,const BigInt& p)
{
  thm_invert(ans.get_thm_t(),x.get_thm_t(),p.get_thm_t());
}

inline int numBits(const BigInt& m)
{
  return m.numBits();
}



inline int numBits(long m)
{
  BigInt& te = BigInt::tmp = m;
  return thm_sizeinbase(te.get_thm_t(),2);
}



inline size_t numBytes(const BigInt& m)
{
  return thm_sizeinbase(m.get_thm_t(),256);
}





inline int probPrime(const BigInt& x)
{
  int ans = thm_probab_prime_p(x.get_thm_t(), max(40, DEFAULT_SECURITY) / 2);
  return ans;
}


inline void bigintFromBytes(BigInt& x,octet* bytes,int len)
{
#ifdef REALLOC_POLICE
  if (rand() % 10000 == 0)
    if (x.get_thm_t()->_mp_alloc < ((len + 7) / 8))
      throw runtime_error("insufficient allocation");
#endif
  thm_import(x.get_thm_t(),len,1,sizeof(octet),0,0,bytes);
}


inline void bytesFromBigint(octet* bytes,const BigInt& x,unsigned int len)
{
  size_t ll = x == 0 ? 0 : numBytes(x);
  if (ll>len)
    { throw invalid_length(); }
  memset(bytes, 0, len - ll);
  size_t l;
  thm_export(bytes + len - ll, &l, 1, sizeof(octet), 0, 0, x.get_thm_t());
  assert(ll == l);
}


inline int isOdd(const BigInt& x)
{
  return thm_odd_p(x.get_thm_t());
}


template<class T>
BigInt sqrRootMod(const T& x);

BigInt powerMod(const BigInt& x,const BigInt& e,const BigInt& p);

// Assume e>=0
int powerMod(int x,int e,int p);

inline int Hwt(int N)
{
  int result=0;
  while(N)
    { result++;
      N&=(N-1);
    }
  return result;
}

template <class T>
int limb_size();

#endif

