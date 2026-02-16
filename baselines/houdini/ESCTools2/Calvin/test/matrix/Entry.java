/* Copyright Hewlett-Packard, 2002 */

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
    
    //@ invariant next_row != null;
    //@ invariant next_col != null;
    //@ invariant prev_row != null;
    //@ invariant prev_col != null;
    //@ invariant val != Rational.zero;
}

