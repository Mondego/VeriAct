class DekkerLockStatic {
    private static boolean a = false;
    private static boolean b = false;
    //@ ghost public static int mx;

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
    /*@ ensures mx == \tid */
    /*@ modifies a, mx */
    /*@ performs (mx) 
        { (\old(mx) == 0 && mx == \tid) || 
	(\old(mx) == \tid && mx == \tid)} */
    static void acquire1() {
	if ( !a ) {
	    while (true) {
		a = true;
		//@ set mx = !b ? \tid : mx;
		if ( !b ) {
		    // assert mx == \tid
		    break;
		}
		a = false;
	    }
	}
    }

    /*@ requires \tid == 1 */
    /*@ requires mx == \tid */
    /*@ ensures mx != \tid */
    /*@ modifies a, mx */
    static void release1() {
	//@ set mx = 0;
	a = false;
    }



    /*@ requires \tid == 2 */
    /*@ ensures mx == \tid */
    /*@ modifies b, mx */
    static void acquire2() {
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
    /*@ modifies b, mx */
    static void release2() {
	//@ set mx = 0;
	b = false;
    }


}
