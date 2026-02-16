/* Copyright Hewlett-Packard, 2002 */

class Dekker {
    static int a = 0;
    static int b = 0;
    static boolean critical1 = false;
    static boolean critical2 = false;
    //@ ghost public static boolean gc1;
    //@ ghost public static boolean gc2;

    /*@ oti \tid == 1 ==> a == \old(a) */
    /*@ oti \tid == 2 ==> b == \old(b) */
    /*@ oti \tid == 1 ==> critical1 == \old(critical1) */
    /*@ oti \tid == 2 ==> critical2 == \old(critical2) */
    /*@ oti \tid == 1 ==> gc1 == \old(gc1) */
    /*@ oti \tid == 2 ==> gc2 == \old(gc2) */
    /*@ global_invariant critical1 ==> a == 1 */
    /*@ global_invariant critical2 ==> b == 1 */
    /*@ global_invariant gc1 ==> a == 1 */
    /*@ global_invariant gc2 ==> b == 1 */
    /*@ global_invariant gc1 ==> !critical2 */
    /*@ global_invariant gc2 ==> !critical1 */
    /*@ global_invariant !(gc1 && gc2) */
    /*@ global_invariant !(critical1 && critical2) */
 

    /*@ requires !critical1 && !critical2 */
    /*@ requires !gc2 && !gc1 */
    /*@ requires \tid == 1 */
    static void thread1() {
	while (true) {
	    a = 1;
	    // a++;  /* another implementation */
	    //@ set gc1 = (b == 0);
	    if (b==0) {
		critical1 = true;
	    } 	  
	    //@ set gc1 = false;
	    critical1 = false;
	    a = 0;
	    // a--;  /* another implementation */
	}
    }
    
    /*@ requires \tid == 2 */
    /*@ requires !critical2 && !critical1 */
    /*@ requires !gc2 && !gc1 */
    static void thread2() {
	while (true) {
	    b = 1;
	    //@ set gc2 = (a == 0);
	    if (a==0) {
		critical2 = true;
	    }
	    //@ set gc2 = false;
	    critical2 = false;
	    b = 0;
	}
    }

    
}
    


 
