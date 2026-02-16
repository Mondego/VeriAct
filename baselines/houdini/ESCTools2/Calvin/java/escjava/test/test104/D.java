/* Copyright Hewlett-Packard, 2002 */

class D {

    //@ ghost /*@ guarded_by mx == \tid */ public Thread mx


    //@ guarded_by mx == \tid
    int x;


    //@ ghost /*@ guarded_by[i] mx == i */ public Thread -> boolean foo

    //@ ghost /*@ guarded_by[i] mx == i */ public int -> boolean boo

    //@ ghost /*@ guarded_by[i] mx == i */ public Thread -> Thread -> boolean goo

    //@ ghost /*@ guarded_by[i][j] mx == i */ public Thread -> boolean moo


    //@ elems_guarded_by[i] x == i
    int[] a;    

    //@ elems_guarded_by[i] mx == i
    int[] b;

    //@ elems_guarded_by[i][j] x == i || x == j
    int[][] c;

    //@ elems_guarded_by[i] mx == i
    int[][] e;
    
}
