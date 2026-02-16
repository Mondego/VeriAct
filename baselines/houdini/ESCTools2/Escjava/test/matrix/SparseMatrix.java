package matrix;

class SparseMatrix
{
    int rows;
    int cols;
    Entry[] row_anchors;
    Entry[] col_anchors;
    int[] row_count;
    int[] col_count;
    
    //@ invariant rows >= 0;
    //@ invariant cols >= 0;
    //@ invariant row_anchors != null;
    //@ invariant col_anchors != null;
    //@ invariant row_count != null;
    //@ invariant col_count != null;
    //@ invariant row_anchors.length == rows;
    //@ invariant col_anchors.length == cols;
    //@ invariant row_count.length == rows;
    //@ invariant col_count.length == cols;
    //@ invariant (\forall int i; i >= 0 && i < rows ==> row_anchors[i] != null);
    //@ invariant (\forall int i; i >= 0 && i < cols ==> col_anchors[i] != null);
    //@ invariant (\forall int i; i >= 0 && i < rows ==> row_anchors[i].row == i && row_anchors[i].col == -1);
    //@ invariant (\forall int i; i >= 0 && i < cols ==> col_anchors[i].col == i && col_anchors[i].row == -1);
    
    // Create all-zero matrix with <m> rows and <n> columns.
    //@ requires m >= 0;
    //@ requires n >= 0;
    SparseMatrix(int m, int n)
    {
	rows = m;
	cols = n;
	row_anchors = new Entry[rows];
	for(int i = 0; i < rows; i++)
	{
	    Entry anchor = new Entry();
	    anchor.row = i;
	    anchor.col = -1;
	    anchor.next_row = anchor;
	    anchor.prev_row = anchor;
	    row_anchors[i] = anchor;
	}
	col_anchors = new Entry[cols];
	for(int i = 0; i < cols; i++)
	{
	    Entry anchor = new Entry();
	    anchor.col = i;
	    anchor.row = -1;
	    anchor.next_col = anchor;
	    anchor.prev_col = anchor;
	    col_anchors[i] = anchor;
	}
	row_count = new int[rows];
	col_count = new int[cols];
    }
    
    // Requires <row,col> entry is zero.
    //@ requires row >= 0 && row < rows;
    //@ requires col >= 0 && col < cols;
    //@ requires val != Rational.zero;
    void add(int row, int col, long val)
    {
	// Make new entry.
	Entry n = new Entry();
	n.row = row;
	n.col = col;
	n.val = val;
	row_count[row]++;
	col_count[col]++;
	
	// Splice into row at correct place.
	Entry anchor = row_anchors[row];
	Entry a = anchor;
	Entry b = anchor.next_row;
	while(b != anchor && col > b.col)
	{
	    a = b;
	    b = b.next_row;
	}
	//X.assert(a.col < col);
	//X.assert(b == anchor || b.col > col);
	n.next_row = b;
	n.prev_row = a;
	a.next_row = n;
	b.prev_row = n;
	
	// Splice into col at correct place.
	anchor = col_anchors[col];
	a = anchor;
	b = anchor.next_col;
	while(b != anchor && row > b.row)
	{
	    a = b;
	    b = b.next_col;
	}
	//X.assert(a.row < row);
	//X.assert(b == anchor || b.row > row);
	n.next_col = b;
	n.prev_col = a;
	a.next_col = n;
	b.prev_col = n;
    }
    
    //@ requires row >= 0 && row < rows;
    //@ requires col >= 0 && col < cols;
    long get(int row, int col)
    {
	// Search down whichever list (row or column) is shorter.
	if (row_count[row] <= col_count[col])
	{
	    Entry anchor = row_anchors[row];
	    for(Entry e = anchor.prev_row; e != anchor; e = e.prev_row)
		if (e.col == col)
		    return e.val;
	}
	else
	{
	    Entry anchor = col_anchors[col];
	    for(Entry e = anchor.prev_col; e != anchor; e = e.prev_col)
		if (e.row == row)
		    return e.val;
	}
	return Rational.zero;
    }
    
    //@ requires row >= 0 && row < rows;
    //@ requires col >= 0 && col < cols;
    void set(int row, int col, long val)
    {
	// Search down whichever list (row or column) is shorter.
	if (row_count[row] <= col_count[col])
	{
	    Entry anchor = row_anchors[row];
	    for(Entry e = anchor.next_row; e != anchor; e = e.next_row)
		if (e.col == col)
		{
		    e.val = val;
		    if (val == Rational.zero)
			splice_out(e);
		    return;
		}
	}
	else
	{
	    Entry anchor = col_anchors[col];
	    for(Entry e = anchor.next_col; e != anchor; e = e.next_col)
		if (e.row == row)
		{
		    e.val = val;
		    if (val == Rational.zero)
			splice_out(e);
		    return;
		}
	}
	if (val == Rational.zero) return;
	add(row, col, val);
    }
    
    // Splice entry <e> out of matrix.
    //@ requires e != null;
    //@ requires e.val == Rational.zero;
    void splice_out(Entry e)
    {
	//X.assert(e.val == Rational.zero);
	
	Entry prev = e.prev_row;
	Entry next = e.next_row;
	prev.next_row = next;
	next.prev_row = prev;
	row_count[e.row]--;
	
	prev = e.prev_col;
	next = e.next_col;
	prev.next_col = next;
	next.prev_col = prev;
	col_count[e.col]--;
    }
    
    //@ requires col >= 0 && col < cols;
    Entry column(int col)
    {
	return col_anchors[col];
    }
    //@ requires row >= 0 && row < rows;
    Entry row(int row)
    {
	return row_anchors[row];
    }
    
    double density()
    {
	if (rows == 0 || cols == 0) return Double.NaN;
	int entries = 0;
	for(int i = 0; i < rows; i++)
	    entries += row_count[i];
	return (double)entries / rows / cols;
    }
    
    void print()
    {
	for(int i = 0; i < rows; i++)
	{
	    int j = 0;
	    Entry anchor = row_anchors[i];
	    for(Entry e = anchor.next_row; e != anchor; e = e.next_row)
	    {
		while(j < e.col)
		{
		    System.out.print("_ ");
		    j++;
		}
		System.out.print(Rational.toString(e.val));
		System.out.print(" ");
		j++;
	    }
	    while(j < cols)
	    {
		System.out.print("_ ");
		j++;
	    }
	    System.out.println();
	}
    }
}
