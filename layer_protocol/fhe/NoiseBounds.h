/*
 * NoiseBound.h
 *
 */

#ifndef FHE_NOISEBOUNDS_H_
#define FHE_NOISEBOUNDS_H_

#include "Math/BigInt.h"

int phi_N(int N);
class FHE_Params;

class SemiHomomorphicNoiseBounds
{
protected:
    static const int FHE_epsilon = 55;

    const BigInt p;
    const int phi_m;
    const int n;
    const int sec;
    int slack;
    mpf_class sigma;

    BigInt B_clean;
    BigInt B_scale;
    BigInt drown;

    mpf_class c1, c2;
    mpf_class V_s;

    void produce_epsilon_constants();

public:
    SemiHomomorphicNoiseBounds(const BigInt& p, int phi_m, int n, int sec,
            int slack, bool extra_h, const FHE_Params& params);
    // with scaling
    BigInt min_p0(const BigInt& p1);
    // without scaling
    BigInt min_p0();
    BigInt min_p0(bool scale, const BigInt& p1) { return scale ? min_p0(p1) : min_p0(); }
    static double min_phi_m(int log_q, double sigma);
    static double min_phi_m(int log_q, const FHE_Params& params);

    BigInt get_B_clean() { return B_clean; }
};

// as per ePrint 2012:642 for slack = 0
class NoiseBounds : public SemiHomomorphicNoiseBounds
{
    BigInt B_KS;

public:
    NoiseBounds(const BigInt& p, int phi_m, int n, int sec, int slack,
            const FHE_Params& params);
    BigInt U1(const BigInt& p0, const BigInt& p1);
    BigInt U2(const BigInt& p0, const BigInt& p1);
    BigInt min_p0(const BigInt& p0, const BigInt& p1);
    BigInt min_p0(const BigInt& p1);
    BigInt min_p1();
    BigInt opt_p1();
    BigInt opt_p0() { return min_p0(opt_p1()); }
    double optimize(int& lg2p0, int& lg2p1);
};

#endif /* FHE_NOISEBOUNDS_H_ */
