/* Copyright Hewlett-Packard, 2002 */

class DekkerLock {
    private boolean a = false;
    private boolean b = false;
    //@ ghost public Thread mx;
    //@ non_null
    public final Thread t1, t2;
    /*@ global_invariant t1 != t2 */

    /*@ global_invariant mx == null || mx == t1 || mx == t2 */
    /*@ global_invariant mx == t1 ==> a */
    /*@ global_invariant mx == t2 ==> b */    

    /*@ oti \tid == t1 ==> a == \old(a) */
    /*@ oti \tid == t2 ==> b == \old(b) */
    /*@ oti (\tid == \old(mx)) == (mx == \tid) */

    /*@ invariant \tid == t1 ==> a == (mx == \tid) */
    /*@ invariant \tid == t2 ==> b == (mx == \tid) */
    

    /*@ requires \tid == t1 */
    /*@ modifies a, mx */
    /*@ performs (mx) 
        { (\old(mx) == null && mx == \tid) || 
	(\old(mx) == \tid && mx == \tid)} */
    void acquire1() {
	if ( !a ) {
	    while (true) {
		a = true;
		//@ set mx = !b ? \tid : mx;
		// Need this assert for the one below
		//@ assert !b ==> mx == \tid
		if ( !b ) {
		    // need this assert or the one
		    // above for "performs"
		    //@ assert mx == \tid
		    break;
		}
		a = false;
	    }
	}
    }

    /*@ requires \tid == t1 */
    /*@ requires mx == \tid */
    /*@ modifies a, mx */
    /*@ performs (mx) { mx == null } */     
    void release1() {
	//@ set mx = null;
	a = false;
    }

    /*@ requires \tid == t2 */
    /*@ modifies b, mx */
    /*@ performs (mx) 
        { (\old(mx) == null && mx == \tid) || 
	(\old(mx) == \tid && mx == \tid)} */
    void acquire2() {
	if ( !b ) {
	    while (true) {
		b = true;
		//@ set mx = !a ? \tid : mx;
		//@ assert !a ==> mx == \tid
		if ( !a ) {
		    break;
		}
		b = false;
	    }
	}
    }

    /*@ requires \tid == t2 */
    /*@ requires mx == \tid */
    /*@ modifies b, mx */
    /*@ performs (mx) { mx == null } */     
    void release2() {
	//@ set mx = null;
	b = false;
    }


}
