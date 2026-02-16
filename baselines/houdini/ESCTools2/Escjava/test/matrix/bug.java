class bug
{
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
}
