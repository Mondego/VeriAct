/* Copyright Hewlett-Packard, 2002 */

package matrix;

/** Represents a matrix of the form:
   1     a1
     1   a2
       1 a3
         a4
         a5 1
         a6   1
*/

class ElementaryMatrix
{
    int col;        // non-identity column
    int n;          // # of nonzeros in column
    long[] entries; // non-zero entries (Rationals)
    int[] rows;     // rows that non-zero entries are in (increasing order)
    
    //@ invariant n >= 0;
    //@ invariant entries != null;
    //@ invariant rows != null;
    
    //@ requires data != null;
    ElementaryMatrix(int col, long[] data)
    {
	n = 0;
	for(int i = 0; i < data.length; i++)
	    if (data[i] != Rational.zero) n++;
	
	entries = new long[n];
	rows = new int[n+1];   // +1 so we can set rows[n] == to a large value when we need a "stop" at end of row array.
	
	int j = 0;
	for(int i = 0; i < data.length; i++)
	    if (data[i] != Rational.zero)
	    {
		entries[j] = data[i];
		rows[j] = i;
		j++;
	    }
	//X.assert(j == n);
	this.col = col;
    }
    //@ requires v != null;
    ElementaryMatrix(int col, SparseVector v)
    {
	n = v.n;
	
	entries = new long[n];
	rows = new int[n+1];   // +1 so we can set rows[n] == to a large value when we need a "stop" at end of row array.
	
	System.arraycopy(v.entries, 0, entries, 0, n);
	System.arraycopy(v.rows, 0, rows, 0, n);
	this.col = col;
    }
    
    /** Apply matrix to a dense vector by multiplying on the left.
        if data = <x1 x2 x3 x4 x5 x6>, the result is
	   data = <x1+a1x4 x2+a2x4 x3+a3x4 a4x4 x5+a5x4 x6+a6x4>. */
    //@ requires data != null;
    void apply_left(long[] data)
    {
	long piv = data[col];                 // x4 in example.
	if (piv == Rational.zero) return;
	
	// Set x4 to 0 so that when we compute x4 + a4*piv, we get a4*piv.
	data[col] = Rational.zero;
	
	for(int i = 0; i < n; i++)
	{
	    int row = rows[i];
	    long entry = entries[i];
	    data[row] = Rational.add(data[row], Rational.mul(entry, piv));
	}
    }
    
    /** Apply matrix to a dense vector by multiplying on the right.
        if data = <x1 x2 x3 x4 x5 x6>, the result is
	   data = <x1 x2 x3 (a1x1+a2x2+...+a6x6) x5 x6>. */
    //@ requires data != null;
    void apply_right(long[] data)
    {
	long v = Rational.zero;
	
	for(int i = 0; i < n; i++)
	{
	    int row = rows[i];
	    long entry = entries[i];
	    v = Rational.add(v, Rational.mul(entry, data[row]));
	}
	data[col] = v;
    }
    
    //@ requires v != null;
    void apply_left(SparseVector v)
    {
	// Cache pointers to matrix and vector data.
	long[] mentries = entries;
	int[] mrows = rows;
	int mn = n;
	long[] ventries = v.entries;
	int[] vrows = v.rows;
	int vn = v.n;
	
	// First, find pivot (x4).
	long piv = Rational.zero;
	for(int i = 0; i < vn; i++)
	    if (vrows[i] == col)
	    {
		piv = ventries[i];
		ventries[i] = Rational.zero;
		break;
	    }
	if (piv == Rational.zero) return;
	
	// Insert "stops".
	mrows[mn] = Integer.MAX_VALUE;
	vrows[vn] = Integer.MAX_VALUE;
	
	// TODO: optimize this allocation away.
	long[] e = new long[mn + vn];
	int[] r = new int[mn + vn];
	int k = 0;
	
	int matrix_idx = 0;
	int vector_idx = 0;
	int matrix_row = mrows[0];
	int vector_row = vrows[0];
	while(true)
	{
	    if (matrix_row == vector_row)
	    {
		if (matrix_row == Integer.MAX_VALUE) break;
		
		// compute x + a * piv
		e[k] = Rational.add(ventries[vector_idx],
				    Rational.mul(mentries[matrix_idx], piv));
		r[k] = matrix_row;
		if (e[k] != Rational.zero)
		    k++;
		matrix_idx++;
		vector_idx++;
		matrix_row = mrows[matrix_idx];
		vector_row = vrows[vector_idx];
	    }
	    else if (matrix_row > vector_row)
	    {
		// a = 0, so just copy x value.
		e[k] = v.entries[vector_idx];
		r[k] = vector_row;
		k++;
		vector_idx++;
		vector_row = vrows[vector_idx];
	    }
	    else
	    {
		// x = 0, so just put a * piv value.
		e[k] = Rational.mul(entries[matrix_idx], piv);
		r[k] = matrix_row;
		k++;
		matrix_idx++;
		matrix_row = mrows[matrix_idx];
	    }
	}
	
	// copy from e & r back into v.
	System.arraycopy(e, 0, ventries, 0, k);
	System.arraycopy(r, 0, vrows, 0, k);
	v.n = k;
    }
    
    //@ requires v != null;
    void apply_right(SparseVector v)
    {
	// Cache pointers to matrix and vector data.
	long[] mentries = entries;
	int[] mrows = rows;
	int mn = n;
	long[] ventries = v.entries;
	int[] vrows = v.rows;
	int vn = v.n;
	
	// Insert "stops".
	mrows[mn] = Integer.MAX_VALUE;
	vrows[vn] = Integer.MAX_VALUE;
	
	// Compute dot product of <v> and matrix column.
	long dot = Rational.zero;
	
	int matrix_idx = 0;
	int vector_idx = 0;
	int matrix_row = mrows[0];
	int vector_row = vrows[0];
	while(true)
	{
	    if (matrix_row == vector_row)
	    {
		if (matrix_row == Integer.MAX_VALUE) break;
		dot = Rational.add(dot, Rational.mul(mentries[matrix_idx],
						     ventries[vector_idx]));
		matrix_idx++;
		vector_idx++;
		matrix_row = mrows[matrix_idx];
		vector_row = vrows[vector_idx];
	    }
	    else if (matrix_row > vector_row)
	    {
		vector_idx++;
		vector_row = vrows[vector_idx];
	    }
	    else
	    {
		matrix_idx++;
		matrix_row = mrows[matrix_idx];
	    }
	}
	
	// Install <dot> in vector at row <col>.
	int i = 0;
	while(col > vrows[i] && i < vn) i++;
	
	if (col == vrows[i])
	{
	    ventries[i] = dot;
	    if (dot == Rational.zero)
	    {
		System.arraycopy(ventries, i + 1, ventries, i, vn - (i + 1));
		System.arraycopy(vrows, i + 1, vrows, i, vn - (i + 1));
		v.n = vn - 1;
	    }
	}
	else // col < vrows[i] or i == vn
	{
	    if (dot != Rational.zero)
	    {
		System.arraycopy(ventries, i, ventries, i + 1, vn - i);
		System.arraycopy(vrows, i, vrows, i + 1, vn - i);
		ventries[i] = dot;
		vrows[i] = col;
		v.n = vn + 1;
	    }
	}
    }
}
