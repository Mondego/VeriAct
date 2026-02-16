/* Copyright Hewlett-Packard, 2002 */


class C2 {

    C2() { } //@ nowarn

    /*@ non_null */ static int [][] a; 
    //@ invariant \nonnullelements(a);
    static int c,d;

    void foo20 () {	
	// loop_invariant \nonnullelements(a);
	//@ loop_predicate \nonnullelements(a);
	for (int i = 0; i < a.length; i++) {

	    //@ assert \nonnullelements(a);

	    // loop_invariant \nonnullelements(a);
	    //@ loop_predicate \nonnullelements(a);
	    for (int j = 0; j < a[i].length; j++) { //@ nowarn IndexNegative, IndexTooBig
		//@ assert \nonnullelements(a);
		a[i][j] = 0; //@ nowarn IndexNegative, IndexTooBig
		//@ assert \nonnullelements(a);
	    }
	}
    }


    void foo21 () {	
	//@ assume 0 <= c && 0 <= d

	// need this
	// loop_invariant c < i && d < a[c].length ==> a[c][d] == 0 

	//@ loop_invariant i >= 0;
	//@ loop_invariant \nonnullelements(a);

	//@ loop_predicate c < i && d < a[c].length ==> a[c][d] == 0 
	// loop_predicate i >= 0;
	// loop_predicate \nonnullelements(a);

	for (int i = 0; i < a.length; i++) {
	    
	    // assert c < i && d < a[c].length ==> a[c][d] == 0 

	    // assert \nonnullelements(a);

	    int j;

	    //@ loop_invariant c < i && d < a[c].length ==> a[c][d] == 0 
	    //@ loop_invariant d < j ==> a[i][d] == 0 
	    //@ loop_invariant \nonnullelements(a);
	    //@ loop_invariant j >= 0;

	    // loop_predicate ((c < i && d < a[c].length) ==> a[c][d] == 0 )
	    // loop_predicate d < j ==> a[i][d] == 0 
	    // loop_predicate c < i, d < a[c].length, a[c][d] == 0 
	    // loop_predicate d < j, a[i][d] == 0 
	    // loop_predicate \nonnullelements(a);
	    // loop_predicate j >= 0;

	    for (j = 0; j < a[i].length; j++) {
		// assert ((c < i && d < a[c].length) ==> a[c][d] == 0 )
		// assert d < j ==> a[i][d] == 0 
		// assert \nonnullelements(a);
		a[i][j] = 0;
		// assert \nonnullelements(a);
	    }
	    // assert c < i && d < a[c].length ==> a[c][d] == 0 
	    // assert d < j ==> a[i][d] == 0 
	    // assert j >= a[i].length
	    //@ assert !(j < a[i].length)
	    // assert d < a[i].length ==> a[i][d] == 0 
	    //@ assert c < i+1 && d < a[c].length ==> a[c][d] == 0 
	}
	
	/*@ assert 0 <= c && c < a.length && 0 <= d && d < a[c].length 
                   ==> a[c][d] == 0 */
    }

    void foo22 () {	
	//@ assume 0 <= c && 0 <= d

	/* loop_predicate 0 <= c, 0 <= d, c < i, c = i, 
	  d < j, d < a[c].length, a[c][d] == 0, 
	  i >= 0
	*/

	// loop_invariant \nonnullelements(a);
	// loop_invariant c < i && d < a[c].length ==> a[c][d] == 0 

	// loop_predicate c < i && d < a[c].length ==> a[c][d] == 0 

	//@ loop_predicate c < i, d < a[c].length, a[c][d] == 0 
	//@ loop_predicate i >= 0;
	//@ loop_predicate \nonnullelements(a);

	for (int i = 0; i < a.length; i++) {

	    //@ assert c < i && d < a[c].length ==> a[c][d] == 0 
	    //@ assert \nonnullelements(a);

	    // loop_invariant c < i && d < a[c].length ==> a[c][d] == 0 
	    // loop_invariant d < j ==> a[i][d] == 0 

	    // loop_predicate c < i && d < a[c].length ==> a[c][d] == 0 
	    // loop_predicate d < j ==> a[i][d] == 0 

	    //@ loop_predicate c < i, d < a[c].length, a[c][d] == 0 
	    //@ loop_predicate d < j, a[i][d] == 0 
	    //@ loop_predicate j >= 0;
	    //@ loop_predicate \nonnullelements(a);

	    for (int j = 0; j < a[i].length; j++) {
		//@ assert d < j ==> a[i][d] == 0 
		//@ assert \nonnullelements(a);
		a[i][j] = 0;
		//@ assert \nonnullelements(a);
	    }
	    //@ assert c < i && d < a[c].length ==> a[c][d] == 0 
	}
	
	/*@ assert 0 <= c && c < a.length && 0 <= d && d < a[c].length 
                   ==> a[c][d] == 0 */
    }

    void foo23 () {	
	// assume 0 <= c && 0 <= d

	// loop_invariant \nonnullelements(a);
	// loop_invariant c < i && d < a[c].length ==> a[c][d] == 0 

	// loop_predicate c < i && d < a[c].length ==> a[c][d] == 0 

	//@ loop_predicate c < i, d < a[c].length, a[c][d] == 0 
	//@ loop_predicate i >= 0;
	//@ loop_predicate \nonnullelements(a);
	//@ loop_predicate 0 <= c, 0 <= d

	for (int i = 0; i < a.length; i++) {

	    //@ assert 0 <= c && 0 <= d && c < i && d < a[c].length ==> a[c][d] == 0 
	    //@ assert \nonnullelements(a);

	    // loop_invariant c < i && d < a[c].length ==> a[c][d] == 0 
	    // loop_invariant d < j ==> a[i][d] == 0 

	    // loop_predicate c < i && d < a[c].length ==> a[c][d] == 0 
	    // loop_predicate d < j ==> a[i][d] == 0 

	    //@ loop_predicate c < i, d < a[c].length, a[c][d] == 0 
	    //@ loop_predicate d < j, a[i][d] == 0 
	    //@ loop_predicate j >= 0;
	    //@ loop_predicate \nonnullelements(a);
	    //@ loop_predicate 0 <= c, 0 <= d

	    for (int j = 0; j < a[i].length; j++) {
		//@ assert 0 <= d && d < j ==> a[i][d] == 0 
		//@ assert \nonnullelements(a);
		a[i][j] = 0;
		//@ assert \nonnullelements(a);
	    }
	    //@ assert 0 <= c && 0 <= d && c < i && d < a[c].length ==> a[c][d] == 0 
	}
	
	/*@ assert 0 <= c && c < a.length && 0 <= d && d < a[c].length 
                   ==> a[c][d] == 0 */
    }

}

