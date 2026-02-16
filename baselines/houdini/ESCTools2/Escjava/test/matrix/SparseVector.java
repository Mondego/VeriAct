package matrix;

class SparseVector
{
    /*@ non_null */ long[] entries;  // storage for non-zero entries.
    /*@ non_null */ int[] rows;      // rows that non-zero entries are in (increasing order)
    int n;           // Actual number of entries.
    int length;      // length of vector (i.e., its size when dense).
    
    //@ invariant entries.length == length;
    //@ invariant rows.length == length + 1;
    //@ invariant length >= 0;
    //@ invariant n >= 0;
    //@ invariant n <= length;
    //@ invariant (\forall int i; i >= 0 && i < n ==> entries[i] != Rational.zero);
    //@ invariant (\forall int i; i >= 0 && i < n ==> rows[i] < length);
    
    /** <len> is the length of the equivalent dense vector.  Vector
	returned is initialized to all zero. */
    //@ requires len >= 0;
    //@ ensures entries != null;
    //@ ensures rows != null;
    //@ ensures entries.length == len;
    //@ ensures rows.length == len+1;
    //@ ensures n == 0;
    //@ ensures length == len;
    SparseVector(int len) {
	entries = new long[len];
	rows = new int[len+1];   // +1 so we can set rows[n] == to a large value when we need a "stop" at end of row array.
	n = 0;
	length = len;
    }
    
    /** set to all zeros */
    //@ ensures n == 0;
    void zero() {
	n = 0;
    }
    
    //@ requires anchor.initialized;
    void copy_from(/*@ non_null */ Entry anchor) {
	int i = 0;
	for(Entry e = anchor.next_col; e != anchor; e = e.next_col) {
	    entries[i] = e.val;
	    rows[i] = e.row;
	    i++;
	}
	n = i;
    }
    
    // NOTE: added these because System.arraycopy is not specified
    //       for copies of arrays with non-Object basetypes.
    //@ requires len >= 0;
    //@ requires ai >= 0;
    //@ requires bi >= 0;
    //@ requires ai+len <= a.length;
    //@ requires bi+len <= b.length;
    //@ ensures (\forall int i; 0 <= i && i < len ==> b[bi+i] == a[ai+i]);
    static void my_arraycopy(/*@ non_null */ int[] a, int ai, 
                             /*@ non_null */ int[] b, int bi, int len) {
	for(int i = 0; i < len; i++)
	    b[bi + i] = a[ai + i];
    }

    //@ requires len >= 0;
    //@ requires ai >= 0;
    //@ requires bi >= 0;
    //@ requires ai+len <= a.length;
    //@ requires bi+len <= b.length;
    //@ ensures (\forall int i; 0 <= i && i < len ==> b[bi+i] == a[ai+i]);
    static void my_arraycopy(/*@ non_null */ long[] a, int ai, 
                             /*@ non_null */ long[] b, int bi, int len) {
	for(int i = 0; i < len; i++)
	    b[bi + i] = a[ai + i];
    }
    
    //@ requires length == v.length;
    void copy_from(/*@ non_null */ SparseVector v) {
	n = v.n;
	my_arraycopy(v.entries, 0, entries, 0, n);
	my_arraycopy(v.rows, 0, rows, 0, n);
    }
    
    //@ requires data.length <= length;
    void copy_from(/*@ non_null */ long[] data) {
	int k = 0;
	for(int i = 0; i < data.length; i++) {
	    if (data[i] != Rational.zero) {
		entries[k] = data[i];
		rows[k] = i;
		k++;
	    }
	}
	n = k;
    }
    
    // Returns entry <row> of vector, and sets that entry to <val>.
    // entry <row> must be nonzero to start, and <val> must be nonzero.
    //@ requires val != Rational.zero;
    //@ requires (\exists int i; i >=0 && i < n && rows[i] == row);
    //@ ensures \result != Rational.zero;
    long swap_entry(int row, long val)
    {
	assert(val != Rational.zero);
	for(int i = 0; i < n; i++)
	    if (rows[i] == row) {
		long ret = entries[i];
		entries[i] = val;
		return ret;
	    }
        assert false;
	//@ nowarn Reachable;
	return 0;
    }
    
    //@ requires row < length;
    long get_entry(int row) {
	for(int i = 0; i < n; i++)
	    if (rows[i] == row)
		return entries[i];
	return Rational.zero;
    }

    //@ requires anchor.initialized;
    long dot(/*@ non_null */ Entry anchor) {
	long dot = Rational.zero;
	int idx = 0;
	Entry e = anchor.next_col;
	while(true) {
	    if (e == anchor || idx == n) break;
	    if (e.row == rows[idx]) {
		dot = Rational.add(dot, Rational.mul(e.val, entries[idx]));
		e = e.next_col;
		idx++;
	    } else if (e.row < rows[idx]) {
		e = e.next_col;
	    } else {
		idx++;
	    }
	}
	return dot;
    }
}
