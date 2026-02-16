package matrix;

class Entry
{
    int row;
    int col;
    Entry next_row;  // Next entry in row
    Entry next_col;  // Next entry in col
    Entry prev_row;  // Previous entry in row
    Entry prev_col;  // Previous entry in col
    long val;
    boolean initialized;

    //@ invariant row >= 0;
    //@ invariant col >= 0;
    //@ invariant initialized ==> (next_row != null);
    //@ invariant initialized ==> (next_col != null);
    //@ invariant initialized ==> (prev_row != null);
    //@ invariant initialized ==> (prev_col != null);
    //@ invariant initialized ==> (val != Rational.zero);
}

