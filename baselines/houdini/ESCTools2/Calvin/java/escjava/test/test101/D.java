/* Copyright Hewlett-Packard, 2002 */

class D {
    /*@ non_null */ static int [][] a; 
    //@ invariant \nonnullelements(a);
    D(){} //@ nowarn

    void foo5 () {	
	int[] b = a[0];
	a[0][0] = 0;
	//@ assert b == a[0];
    }
}
