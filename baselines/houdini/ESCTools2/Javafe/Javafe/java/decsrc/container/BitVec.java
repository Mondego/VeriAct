package decsrc.container;

/**
 * A set of integers that can hold integers in the range
 * <code>[0..N-1]</code> for any <code>N >= 0</code>.  Requires space
 * <i>O(N)</i>.
 *
 * <p> The <code>BitVec</code> class is thread safe, but instances of
 * <code>BitVec</code> are not.  That is, two different threads may
 * safely access two different instances of <code>BitVec</code>
 * concurrently, but they may not safely access the same instance of
 * <code>BitVec</code> without synchronization external to
 * <code>BitVec</code>. </p>
 *
 * <p> The iteration provided by the <pre>next/get</pre> methods of
 * BitVector has a stronger specification.  See the specification of
 * <pre>next</pre> for more details. </p>
 */

final public class BitVec extends IntSet
{
    private static final int BITS = 64;
    private static final int MASK = BITS - 1;
    private static final int LGBITS = 6;
    
    private /*@non_null*/long[] words;
    //@ private invariant num >= 0;
    private int    num;

    // If num is not a multiple of 64, the extra bits in words are zero.

    // BitVec specific operations

    /**
     * Creates an empty bitvector of length "n".
     */
    //@ private normal_behavior
    //@ requires n >= 0;
    //@ ensures num == n;
    //@ ensures words != null;
    //@ ensures words.length == (n + BITS - 1) >> LGBITS;
    public BitVec(int n) {
	super();
	num = n;
        //@ assert ((n + BITS - 1) >> LGBITS) > 0;
	words = new long[(n + BITS - 1) >> LGBITS];
    }

    /** post(this) = [0,this.length()) */
    //@ private normal_behavior
    //@ requires words != null;
    public void set_all() {
	long[] w = words;
	int n = w.length;
	for (int i = 0; i < n - 1; i++) {
	    w[i] = -1;
	}
	// Set last word correctly, so that extra bits (if any) are zero.
	if ((num & MASK) != 0)
	    w[n - 1] = (1L << num) - 1;
	else
	    w[n - 1] = -1;
    }

    /**
     * Ensures that bitvector length is at least "n".
     */
    //@ private normal_behavior
    //@ requires n >= 0;
    //@ ensures num >= n;
    public void ensure(int n) {
	if (n > num) {
	    int new_num = n * 2;
	    int old_count = (num + BITS - 1) >> LGBITS;
	    int new_count = (new_num + BITS - 1) >> LGBITS;
	    long[] copy = new long[new_count];
	    for (int i = 0; i < old_count; i++) {
		copy[i] = words[i];
	    }
	    words = copy;
	    num = new_num;
	}
    }

    // Implementations of supercall operations

    //@ also
    //@ ensures \result == 0;
    public int min_allowed() {
	return 0;
    }

    //@ also
    //@ private normal_behavior
    //@ ensures \result == num - 1;
    public int max_allowed() {
	return num - 1;
    }

    public boolean contains(int i) {
	// Note: only the low six bits of i are used in (1L << i)
	return ((words[i >> LGBITS] & (1L << i)) != 0);
    }

    public boolean insert(int i) {
	long w1 = words[i >> LGBITS];
	long w2 = (w1 | (1L << i));
	words[i >> LGBITS] = w2;
	return (w1 != w2);
    }

    public boolean remove(int i) {
	long w1 = words[i >> LGBITS];
	long w2 = (w1 & ~(1L << i));
	words[i >> LGBITS] = w2;
	return (w1 != w2);
    }

    public void clear() {
	long[] w = words;
	int n = w.length;
	for (int i = 0; i < n; i++) {
	    w[i] = 0;
	}
    }

    public void copy_from(/*@non_null*/IntSet x) {
	if (x instanceof BitVec) {
	    copy_from((BitVec) x);
	} else {
	    super.copy_from(x);
	}
    }

    public void copy_from(/*@non_null*/BitVec x) {
	long[] xw = x.words;
	long[] w = words;
	int n = xw.length;
	int m = w.length;
	for (int i = 0; i < n; i++) {
	    w[i] = xw[i];
	}
	for (int i = n; i < m; i++) {
	    w[i] = 0;
	}
    }

    public boolean union(/*@non_null*/IntSet x) {
	if (x instanceof BitVec) {
	    return union((BitVec) x);
	} else {
	    return super.union(x);
	}
    }

    public boolean union(/*@non_null*/BitVec x) {
	long[] xw = x.words;
	long[] w = words;
	int n = xw.length;
	boolean changed = false;
	for (int i = 0; i < n; i++) {
	    long w1 = w[i];
	    long w2 = w1 | xw[i];
	    w[i] = w2;
	    changed |= (w1 != w2);
	}
	return changed;
    }

    public boolean intersect(/*@non_null*/IntSet x) {
	if (x instanceof BitVec) {
	    return intersect((BitVec) x);
	} else {
	    boolean changed = false;
	    int iter = -1;
	    while ((iter = next(iter)) >= 0) {
		if (!x.contains(iter)) {
		    remove(iter);
		    changed = true;
		}
	    }
	    return changed;
	}
    }

    public boolean intersect(/*@non_null*/BitVec x) {
	long[] xw = x.words;
	long[] w = words;
	int n = xw.length;
	boolean changed = false;
	for (int i = 0; i < n; i++) {
	    long w1 = w[i];
	    long w2 = w1 & xw[i];
	    w[i] = w2;
	    changed |= (w1 != w2);
	}
	return changed;
    }

    public boolean subtract(/*@non_null*/IntSet x) {
	if (x instanceof BitVec) {
	    return subtract((BitVec) x);
	} else {
	    boolean changed = false;
	    int iter = -1;
	    while ((iter = next(iter)) >= 0) {
		if (x.contains(iter)) {
		    remove(iter);
		    changed = true;
		}
	    }
	    return changed;
	}
    }

    public boolean subtract(/*@non_null*/BitVec x) {
	long[] xw = x.words;
	long[] w = words;
	int n = xw.length;
	boolean changed = false;
	for (int i = 0; i < n; i++) {
	    long w1 = w[i];
	    long w2 = w1 & ~xw[i];
	    w[i] = w2;
	    changed |= (w1 != w2);
	}
	return changed;
    }

    public boolean is_subset_of(/*@non_null*/IntSet x) {
	if (x instanceof BitVec) {
	    return is_subset_of((BitVec) x);
	} else {
	    return super.is_subset_of(x);
	}
    }

    public boolean is_subset_of(/*@non_null*/BitVec x) {
	long[] xw = x.words;
	long[] w = words;
	int n = xw.length;
	for (int i = 0; i < n; i++) {
	    if ((w[i] & ~xw[i]) != 0) {
		return false;
	    }
	}
	return true;
    }

    // The top six bits of "2^k * lsb_mult" are different for each
    // possible value of "k" in the range "[0,63]".
    // Furthermore, "lsb_table[(2^k * mult) >>> 58] == k".
    private static final long lsb_mult = 0x07edd5e59a4e28c2L;
    private static final /*@non_null*/byte lsb_table[] = lsb_make_table();

    // Build "lsb_table" and check that there are no duplicates
    private static /*@non_null*/byte[] lsb_make_table() {
	long mult = lsb_mult;
	byte[] table = new byte[64];
	for (int i = 0; i < 64; i++) {
	    table[i] = -1;
	}
	for (int i = 0; i < 64; i++) {
	    long v = 1L << i;
	    int x = (int) ((mult * v) >>> 58);
	    if (table[x] != -1) {
		throw new RuntimeException("bad multiplier for lsb table");
	    }
	    table[x] = (byte) i;
	}
	return table;
    }

    /** Return the smallest index "x" such that "x > iter" and an
	element exists at index "x".  Return -1 if no such index
	exists.  Requires "iter" is in the range "[-1,max_allowed]".
	<p>
	Note that this method has a stronger specification than IntSet.next:
	<p>
	<ul>
	<li> The elements yielded by the iteration are in increasing order.
	<li> The bitvector can be mutated during iteration.
	<li> get(i) == i 
	</ul>
	Therefore <pre>s.next(-1)</pre> returns either -1, or the smallest
	element of <pre>s</pre>.
    */
    public int next(int iter) {
	int i = iter + 1;
	if (i < 0) throw new IllegalArgumentException("negative index");
	int n = num;
	if (i >= n) return -1;
	int w = i >> LGBITS;
	long[] data = words;

	// Mask out all bits numbered less than "i" and then start searching
	long word = data[w] & ~((1L << i) - 1);
	int result = w << LGBITS;
	while (word == 0) {
	    w++;
	    result += BITS;
	    if (result >= n) return -1;
	    word = data[w];
	}

	long x = word & ~(word - 1);
	int pos = lsb_table[(int)((x * lsb_mult) >>> 58)];
	return result + pos;
    }

    public int get(int i) {
	return i;
    }

    /** Return number of set entries in the bitvector. */
    public int count() {
	byte[] ctable = count_table;
	long[] wlist = words;
	int n = wlist.length;
	int result = 0;
	for (int i = 0; i < n; i++) {
	    long w = wlist[i];
	    while (w != 0) {
		result += (int) ctable[(int) (w & 0xffl)];
		w >>>= 8;
	    }
	}
	return result;
    }

    // Map from byte to number of bits set in byte
    private static final /*@non_null*/byte[] count_table = build_count_table();
    private static /*@non_null*/byte[] build_count_table() {
	byte[] ctable = new byte[256];
	for (int i = 0; i < 256; i++) {
	    int n = 0;
	    int j = i;
	    while (j != 0) {
		n += j & 1;
		j >>>= 1;
	    }
	    ctable[i] = (byte) n;
	}
	return ctable;
    }
}
