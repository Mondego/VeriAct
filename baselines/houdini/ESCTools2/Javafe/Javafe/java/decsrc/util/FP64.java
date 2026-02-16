// Fingerprinting code from Andrei Broder
package decsrc.util;

/** This file provides procedures that construct fingerprints of
    strings of bytes via operations in GF[2^64].  GF[64] is represented
    as the set polynomials of degree 64 with coefficients in Z(2),
    modulo an irreducible polynomial P of degree 64.  The computer
    internal representation is a 64 bit long word.
    
    Let g(S) be the string obtained from S by prepending the byte 0x80
    and appending eight 0x00 bytes.  Let f(S) be the polynomial
    associated to the string g(S) viewed as a polynomial with
    coefficients in the field Z(2). The fingerprint of S simply the
    value f(S) modulo P.

    The irreducible polynomial p used as a modulus is

    <pre>
    *       3    7    11    13    16    19    20    24    26    28
    *  1 + x  + x  + x   + x   + x   + x   + x   + x   + x   + x
<p>
    *     29    30    36    37    38    41    42    45    46    48
    *  + x   + x   + x   + x   + x   + x   + x   + x   + x   + x
<p>
    *     50    51    52    54    56    57    59    61    62    64
    *  + x   + x   + x   + x   + x   + x   + x   + x   + x   + x
    </pre>
     */
final public class FP64 {
    // Useful constants
    final private static long fprint_zero = 0L;
    final private static long fprint_p = 0x911498AE0E66BAD6L;	// Polynomial
    final private static long fprint_one = 0x8000000000000000L;
    final private static long fprint_x63 = 0x1L;

    /** Fingerprint for the empty byte sequence. */
    public final static long empty = fprint_p;

    /** Returns the fingerprint formed by adding the low eight
	bits of "b" to "f". */
    public final static long extend_byte(long f, int b) {
	return ((f >>> 8) ^ (ByteModTable_7[((int)(f & 0xff)) ^ (b & 0xff)]));
    }

    /** Returns the fingerprint formed by adding "buf[i..i+n-1]" to "f". */
    public final static long extend(long f, byte[] buf, int i, int n) {
	while (i < n) {
	    f = extend_byte(f, buf[i]);
	    i++;
	}
	return f;
    }

    /** Returns the fingerprint formed by adding "ivalue" to "f". */
    public final static long extend_int(long f, int ivalue) {
	for (int i = 0; i < 4; i++) {
	    f = extend_byte(f, ivalue);
	    ivalue >>= 8;
	}
	return f;
    }

    /** Returns the fingerprint formed by adding "lvalue" to "f". */
    public final static long extend_long(long f, int lvalue) {
	for (int i = 0; i < 8; i++) {
	    f = extend_byte(f, (int)lvalue);
	    lvalue >>= 8;
	}
	return f;
    }

    /** Returns the fingerprint formed by adding "s" to "f". */
    public final static long extend_string(long f, String s) {
	int n = s.length();
	for (int i = 0; i < n; i++) {
	    int x = (int) s.charAt(i);
	    f = extend_byte(f, x);
	    f = extend_byte(f, x >> 8);
	}
	return f;
    }

    /* These are the tables used for computing fingerprints.  The
       ByteModTable could be hardwired.  Note that since we just
       extend a byte at a time, we need just "ByteModeTable[7]". */
    private final static long[] ByteModTable_7 = make_table();

    private static long[] make_table() {
	// Maximum power needed == 127-7*8 == 127 - 56 == 71
	int plength = 72;
	long[] PowerTable = new long[plength];

	long t = fprint_one;
	for (int i = 0; i < plength; i++) {
	    PowerTable[i] = t;
	    //System.out.println("pow[" + i + "] = " + Long.toHexString(t));

	    // t = t * x
	    long mask = ((t & fprint_x63) != 0) ? fprint_p : 0;
	    t = (t >>> 1) ^ mask;
	}

	// Just need the 7th iteration of the ByteModTable initialization code
	long[] table_7 = new long[256];
	for (int j = 0; j <= 255; j++) {
	    long v = fprint_zero;
	    for (int k = 0; k <= 7; k++) {
		if ((j & (1L << k)) != 0) {
		    v ^= PowerTable[127-7*8-k];
		}
	    }
	    table_7[j] = v;
	    //System.out.println("mod7[" + j + "] = " + Long.toHexString(v));
	}

	return table_7;
    }
}
