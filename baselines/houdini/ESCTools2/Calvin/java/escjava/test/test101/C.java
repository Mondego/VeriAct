/* Copyright Hewlett-Packard, 2002 */

class C0 {

    C0() { } //@ nowarn

    /*@ non_null */ static int [] a; 
    static int c;

    void foo000 () {	
	// Use these constants to check that we get info
	// from loop context
	int zero1 = 0, zero2 = 0;

	//@ assume 0<=c
	/*@ loop_predicate c < i, a[c] == 0, i < 0 */
	for (int i = 0; i < a.length; i++) {
	    a[i] = zero1;
	}
	
	/*@ assert 0 <= c && c < a.length ==> a[c] == zero2 */
    }
    void foo00 () {	
	// Use these constants to check that we get info
	// from loop context
	int zero1 = 0, zero2 = 0;

	/*@ loop_predicate 0 <= c, c < i, a[c] == 0, i < 0 */
	for (int i = 0; i < a.length; i++) {
	    a[i] = zero1;
	}
	
	/*@ assert 0 <= c && c < a.length ==> a[c] == zero2 */
    }
}


class C1 {  // Tests 2d arrays

    C1() { } //@ nowarn

    /*@ non_null */ static int [][] a; 
    //@ invariant \nonnullelements(a);
    static int c,d;

    void foo10 () {	
	//@ assume a.length >0
	//@ assume a[0].length >0

	int[] b = a[0];
	a[0][0] = 0;
	//@ assert b == a[0];
    }

    void foo11 () {	
	//@ assume a.length >0
	//@ assume a[0].length >0

	//@ loop_invariant a[0] == \old(a[0]);
	while(true) {
	    a[0][0] = 0;
	}
    }

    void foo12 () {	
	//@ assume 0 <= c && c < a.length

	//@ loop_invariant j >= 0;
	//@ loop_invariant a[c] == \old(a[c]);
	//@ loop_invariant a[0] == \old(a[0]);
	//@ loop_invariant \nonnullelements(a);

	for (int j = 0; j < a[c].length; j++) {
	    a[c][j] = 0;
	}
	
    }

    void foo13 () {	
	//@ assume 0 <= c && c < a.length

	// loop_invariant a[c] == \old(a[c]);
	// loop_invariant a[0] == \old(a[0]);
	// loop_invariant a[c] != a ;

	//@ loop_invariant 0 <= d && d < j ==> a[c][d] == 0 
	//@ loop_invariant j >= 0;
	//@ loop_invariant \nonnullelements(a);
	for (int j = 0; j < a[c].length; j++) {
	    //@ assert 0 <= d && d < j ==> a[c][d] == 0 
	    //@ assert j >= 0;
	    //@ assert \nonnullelements(a);
	    a[c][j] = 0;
	    //@ assert \nonnullelements(a);
	}
	
	/*@ assert 0 <= c && c < a.length && 0 <= d && d < a[c].length 
	  ==> a[c][d] == 0 */
    }

    void foo14 () {	

	// loop_invariant \nonnullelements(a);
	//@ loop_predicate \nonnullelements(a);
	while(true) {
	    //@ assert \nonnullelements(a);
	    a[0][0] = 0;  //@ nowarn IndexTooBig
	    //@ assert \nonnullelements(a);
	}
    }

    void foo15 () {	

	//@ assert \nonnullelements(a);

	// loop_invariant \nonnullelements(a);

	//@ loop_predicate \nonnullelements(a);
	while(true) {
	    //@ assert \nonnullelements(a);
	    a[0][0] = 0;  //@ nowarn IndexTooBig
	    //@ assert \nonnullelements(a);
	}
    }

    void foo16 () {	
	//@ assume 0 <= c && c < a.length

	// loop_invariant 0 <= d && d < j ==> a[c][d] == 0 
	// loop_invariant \nonnullelements(a);

	//@ loop_predicate 0 <= d, d < j, a[c][d] == 0 
	//@ loop_predicate j >= 0;
	//@ loop_predicate \nonnullelements(a);
	for (int j = 0; j < a[c].length; j++) {
	    // assert 0 <= d && d < j ==> a[c][d] == 0 
	    // assert j >= 0;
	    // assert \nonnullelements(a);
	    a[c][j] = 0;
	    // assert \nonnullelements(a);
	}
	
	/*@ assert 0 <= c && c < a.length && 0 <= d && d < a[c].length 
	  ==> a[c][d] == 0 */
    }

}
