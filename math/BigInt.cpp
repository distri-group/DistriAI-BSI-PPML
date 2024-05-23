
#include "BigInt.h"
#include "gfp.h"
#include "gfpvar.h"
#include "Integer.h"
#include "Z2k.h"
#include "Z2k.hpp"
#include "GC/Clear.h"
#include "Tools/Exceptions.h"

#include "BigInt.hpp"

thread_local BigInt BigInt::tmp = 0;
thread_local BigInt BigInt::tmp2 = 0;
thread_local gmp_random BigInt::random;


BigInt powerMod(const BigInt& x,const BigInt& e,const BigInt& p)
{
  BigInt ans;
  if (e>=0)
    { thm_powm(ans.get_thm_t(),x.get_thm_t(),e.get_thm_t(),p.get_thm_t()); }
  else
    { BigInt xi,ei=-e;
      invMod(xi,x,p);
      thm_powm(ans.get_thm_t(),xi.get_thm_t(),ei.get_thm_t(),p.get_thm_t()); 
    }
      
  return ans;
}


int powerMod(int x,int e,int p)
{
  if (e==1) { return x; }
  if (e==0) { return 1; }
  if (e<0)
     { throw not_implemented(); }
   int t=x,ans=1;
   while (e!=0)
     { if ((e&1)==1) { ans=(ans*t)%p; }
       e>>=1;
       t=(t*t)%p;
     }
  return ans;
}


size_t BigInt::report_size(ReportType type) const
{
  size_t res = 0;
  if (type != MINIMAL)
    res += sizeof(*this);
  if (type == CAPACITY)
    res += get_thm_t()->_mp_alloc * sizeof(mp_limb_t);
  else if (type == USED)
    res += abs(get_thm_t()->_mp_size) * sizeof(mp_limb_t);
  else if (type == MINIMAL)
    res += 5 + numBytes(*this);
  return res;
}

template <>
int limb_size<BigInt>()
{
  return 64;
}

template <>
int limb_size<int>()
{
  // doesn't matter
  return 0;
}

BigInt::BigInt(const Integer& x) : BigInt(SignedZ2<64>(x))
{
}


BigInt::BigInt(const GC::Clear& x) : BigInt(SignedZ2<64>(x))
{
}

BigInt::BigInt(const mp_limb_t* data, size_t n_limbs)
{
  thm_import(get_thm_t(), n_limbs, -1, 8, -1, 0, data);
}

string to_string(const BigInt& x)
{
  stringstream ss;
  ss << x;
  return ss.str();
}

#ifdef REALLOC_POLICE
void BigInt::lottery()
{
  if (rand() % 1000 == 0)
    if (rand() % 1000 == 0)
      throw runtime_error("much deallocation");
}
#endif
