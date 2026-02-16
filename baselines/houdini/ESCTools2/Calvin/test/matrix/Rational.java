/* Copyright Hewlett-Packard, 2002 */

package matrix;

abstract class Rational
{
    // Rationals are encoded in a long using the (signed) high 32 bits for
    // the numerator and the (unsigned) low 32 bits for the denominator.
    
    // Invariants:
    // 1: denom > 0
    // 2: gcd(|num|, denom) = 1
    
    // Comparison with (==, !=) works correctly on the corresponding
    // longwords because the representation of any rational is unique.
    
    // Comparison with Rational.zero (==, !=, <, <=, >, >=) works correctly
    // because zero (== 1 as a long) has the smallest positive representation.
    // All other positive rationals have a larger representation, and all
    // negative rationals have a smaller representation.
    
    // Note: comparison of two Rationals both of which are not zero will
    // not work correctly with (<, <=, >, >=).  You must use the lt, le,
    // gt, ge functions instead.
    
    // Some precomputed rationals.
    static final long zero = newint(0);
    static final long one = newint(1);
    static final long negone = newint(-1);
    static final long max = newint(Integer.MAX_VALUE);
    
    static long newrat(int num, int denom)
    {
	long n = num;
	long d = denom;
	if (d < 0) { n = -n; d = -d; }
	long g = gcd(n, d);
	if (g > 1)
	{
	    n /= g;
	    d /= g;
	}
	return (n << 32) | d;
    }
    static long newint(int n)
    {
	return ((long)n << 32) | 1;
    }
    static boolean is_int(long r)
    {
	long d = r & 0xFFFFFFFFL;
	return d == 1;
    }
    
    static long mul(long r1, long r2)
    {
	long n1 = r1 >> 32;
	long n2 = r2 >> 32;
	long d1 = r1 & 0xFFFFFFFFL;
	long d2 = r2 & 0xFFFFFFFFL;
	long n = n1 * n2;
	long d = d1 * d2;
	long g = gcd(n, d);
	if (g > 1)
	{
	    n /= g;
	    d /= g;
	}
	return (n << 32) | d;
    }
    static long div(long r1, long r2) throws ArithmeticException
    {
	long n1 = r1 >> 32;
	long n2 = r2 >> 32;
	long d1 = r1 & 0xFFFFFFFFL;
	long d2 = r2 & 0xFFFFFFFFL;
	if (n2 == 0) { System.out.println(toString(r2)); throw new ArithmeticException(); }
	long n = n1 * d2;
	long d = d1 * n2;
	if (n2 < 0) { n = -n; d = -d; }
	long g = gcd(n, d);
	if (g > 1)
	{
	    n /= g;
	    d /= g;
	}
	return (n << 32) | d;
    }
//     static long neg(long r)
//     {
// 	long n = r >> 32;
// 	long d = r & 0xFFFFFFFFL;
// 	n = -n;
// 	return (n << 32) | d;
//     }
    static long neg(long r)
    {
	return (r ^ 0xFFFFFFFF00000000L) + 0x100000000L;
    }
    static long add(long r1, long r2)
    {
	long n1 = r1 >> 32;
	long n2 = r2 >> 32;
	long d1 = r1 & 0xFFFFFFFFL;
	long d2 = r2 & 0xFFFFFFFFL;
	long n, d;
	if (d1 == d2)
	{
	    n = n1 + n2;
	    d = d1;
	}
	else
	{
	    n = n1 * d2 + n2 * d1;
	    d = d1 * d2;
	}
	long g = gcd(n, d);
	if (g > 1)
	{
	    n /= g;
	    d /= g;
	}
	return (n << 32) | d;
    }
    static long sub(long r1, long r2)
    {
	long n1 = r1 >> 32;
	long n2 = r2 >> 32;
	long d1 = r1 & 0xFFFFFFFFL;
	long d2 = r2 & 0xFFFFFFFFL;
	long n, d;
	if (d1 == d2)
	{
	    n = n1 - n2;
	    d = d1;
	}
	else
	{
	    n = n1 * d2 - n2 * d1;
	    d = d1 * d2;
	}
	long g = gcd(n, d);
	if (g > 1)
	{
	    n /= g;
	    d /= g;
	}
	return (n << 32) | d;
    }
    static boolean lt(long r1, long r2)
    {
	long n1 = r1 >> 32;
	long n2 = r2 >> 32;
	long d1 = r1 & 0xFFFFFFFFL;
	long d2 = r2 & 0xFFFFFFFFL;
	return n1 * d2 < n2 * d1;
    }
    static boolean gt(long r1, long r2)
    {
	long n1 = r1 >> 32;
	long n2 = r2 >> 32;
	long d1 = r1 & 0xFFFFFFFFL;
	long d2 = r2 & 0xFFFFFFFFL;
	return n1 * d2 > n2 * d1;
    }
    static boolean le(long r1, long r2)
    {
	long n1 = r1 >> 32;
	long n2 = r2 >> 32;
	long d1 = r1 & 0xFFFFFFFFL;
	long d2 = r2 & 0xFFFFFFFFL;
	return n1 * d2 <= n2 * d1;
    }
    static boolean ge(long r1, long r2)
    {
	long n1 = r1 >> 32;
	long n2 = r2 >> 32;
	long d1 = r1 & 0xFFFFFFFFL;
	long d2 = r2 & 0xFFFFFFFFL;
	return n1 * d2 >= n2 * d1;
    }
    
    // Requires b > 0.
    // optimized for b > |a|, which for the way it is called, corresponds
    // to rationals between 0 and 1.
    static private long gcd2(long a, long b)
    {
	if (a < 0) a = -a;
	
	if (b < a)
	{
	    long tmp = a;
	    a = b;
	    b = tmp;
	}
	while(a != 0)         // Invariant: b >= a
	{
	    long tmp = a;
	    if (b < (a<<1))
		a = b - a;    // optimization - no divide
	    else
		a = b % a;
	    b = tmp;
	}
	return b;
    }
    
    // Requires b > 0.
    // binary gcd algorithm
    static private long gcd(long a, long b)
    {
	if (a < 0) a = -a;
	
	if (a == 0 || b == 0) return a | b;
	
	// Make both <a> and <b> odd by removing factors of 2.  Keep track
	// of the number of factors of 2 in each of <a> and <b>, and compute
	// the minimum, which is the number of factors of 2 in the gcd.
	int sa = 0;
	while((a & 1) == 0)
	{
	    a >>= 1;
	    sa++;
	}
	int sb = 0;
	while((b & 1) == 0)
	{
	    b >>= 1;
	    sb++;
	}
	int shifts = sa < sb ? sa : sb;
	
	// Both <a> and <b> are now odd.
	while(true)
	{
	    if (a == b)
		return a << shifts;
	    if (a > b)
	    {
		a -= b;                // Makes a even.
		a >>= 1;               // Removes last 0.
		while((a & 1) == 0)    // Remove additional zeros (if any).
		    a >>= 1;
	    }
	    else  // a < b
	    {
		b -= a;
		b >>= 1;
		while((b & 1) == 0)
		    b >>= 1;
	    }
	}
    }
    
    static String toString(long r)
    {
	long n = r >> 32;
	long d = r & 0xFFFFFFFFL;
	if (d == 1)
	    return String.valueOf(n);
	else
	    return String.valueOf(n) + "/" + String.valueOf(d);
    }
}

