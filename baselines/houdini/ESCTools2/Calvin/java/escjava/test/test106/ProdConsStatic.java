import java.lang.Math;

/** ProdCons is a producer-consumer class that
 ** implements a producer-consumer system. The
 ** central data structure is a stack, represented
 ** by a top, bottom, and a buffer. Access to these
 ** elements is protected using a so-called "Dekker"
 ** lock, implemented by the DekkerLock class.
 ** 
 ** There are two threads in the system. Thread 1
 ** is a producer thread, and executes the "put()" 
 ** method. Thread 2 is a consumer thread and executes
 ** the "get()" method.
 **/
class ProdConsStatic {
    static final int BUF_SIZE = 10;
    static int top = 0; /*@ invariant 
			  top >= 0 && top <= BUF_SIZE */
    static /*@ non_null */ int[] buf = 
	new int[BUF_SIZE];


    // ghost public int mx;

    /*** These ghost variables, OTIs and global 
	 invariants are derived from the DekkerLock class
    ***/
    /* global_invariant mx == 0 || mx == 1 || mx == 2 */
    /* oti \tid == \old(mx) ==> \old(mx) == mx */
    /* oti \tid != \old(mx) ==> \tid != mx */
    /*---------- end DekkerLock invariants ---------*/    


    /*@ oti 
      DekkerLockStatic.mx == \tid ==> \old(top) == top */
    /*@ oti 
      DekkerLockStatic.mx == \tid ==> \old(buf) == buf */
    /* oti 
      \old(DekkerLockStatic.mx) == \tid ==> 
      \old(DekkerLockStatic.mx) == DekkerLockStatic.mx */
    /*@ global_invariant top >= 0 && top <= BUF_SIZE */
      

    /*@ requires \tid == 1 */
    /*@ requires datum >= 0 */
    void put(int datum) {
	while (true) {
	    DekkerLockStatic.acquire1();
	    // assume DekkerLockStatic.mx == 0;
	    // set DekkerLockStatic.mx = \tid;
	    if (top < BUF_SIZE) {
		buf[top] = datum;
		top++;
		// set DekkerLockStatic.mx = 0;
		DekkerLockStatic.release1(); 
		/* might be interesting
		   to leave this out and 
		   see what the checker
		   turns up */
		break;
	    }
	    // set DekkerLockStatic.mx = 0;
	    DekkerLockStatic.release1();
	}
    }


    /*@ requires \tid == 2 */
    /*@ ensures \result >= 0 */
    int get() {
	int retVal = -1;
	while (true) {
	    DekkerLockStatic.acquire2();
	    if (top > 0) {
		retVal = buf[top-1];
		top--;
		DekkerLockStatic.release2();
		break;
	    }
	    DekkerLockStatic.release2();
	}
	return retVal;
    }


}
