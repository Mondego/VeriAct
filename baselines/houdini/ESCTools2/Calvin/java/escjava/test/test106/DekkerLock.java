class DekkerLock {
    private boolean a = false;
    private boolean b = false;
    //@ ghost public int mx;

    /*@ global_invariant mx == 0 || mx == 1 || mx == 2 */
    /*@ global_invariant mx == 1 ==> a */
    /*@ global_invariant mx == 2 ==> b */    

    /*@ oti \tid == 1 ==> a == \old(a) */
    /*@ oti \tid == 2 ==> b == \old(b) */
    /*@ oti \tid == \old(mx) ==> \old(mx) == mx */
    /*@ oti \tid != \old(mx) ==> \tid != mx */

    /*@ invariant \tid == 1 ==> a == (mx == \tid) */
    /*@ invariant \tid == 2 ==> b == (mx == \tid) */
    

    /*@ requires \tid == 1 */
    /*@ modifies mx, a */
    /*@ performs (mx) 
        { (\old(mx) == 0 && mx == \tid) || 
	(\old(mx) == \tid && mx == \tid)} */
    void acquire1() {
	if ( !a ) {
	    while (true) {
		a = true;
		//@ set mx = !b ? \tid : mx;
		if ( !b ) {
		    break;
		}
		a = false;
	    }
	}
    }

    /*@ requires \tid == 1 */
    /*@ requires mx == \tid */
    /*@ modifies mx, a */
    /*@ performs (mx) { mx == 0 } */     
    void release1() {
	//@ set mx = 0;
	a = false;
    }

    /*@ requires \tid == 2 */
    /*@ modifies mx, b */
    /*@ performs (mx) 
        { (\old(mx) == 0 && mx == \tid) || 
	(\old(mx) == \tid && mx == \tid)} */
    void acquire2() {
	if ( !b ) {
	    while (true) {
		b = true;
		//@ set mx = !a ? \tid : mx;
		if ( !a ) {
		    break;
		}
		b = false;
	    }
	}
    }

    /*@ requires \tid == 2 */
    /*@ requires mx == \tid */
    /*@ modifies mx, b */
    /*@ performs (mx) { mx == 0 } */     
    void release2() {
	//@ set mx = 0;
	b = false;
    }


}
