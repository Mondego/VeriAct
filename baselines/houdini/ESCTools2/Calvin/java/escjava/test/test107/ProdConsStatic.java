/* Copyright Hewlett-Packard, 2002 */

import java.lang.Math;

/** ProdCons is a producer-consumer class that
 ** implements a producer-consumer system. The
 ** central data structure is a stack, represented
 ** by a top, bottom, and a buffer. Access to these
 ** elements is protected using a so-called "Dekker"
 ** lock, implemented by the DekkerLockStatic class.
 ** 
 ** There are two threads in the system. Thread 1
 ** is a producer thread, and executes the "put()" 
 ** method. Thread 2 is a consumer thread and executes
 ** the "get()" method.
 **/
class ProdConsStatic {
    static final int BUF_SIZE = 10;
    static /*@ non_null */ int[] buf = 
	new int[BUF_SIZE];
    /*@ global_invariant (\forall int i; 
      (i >= 0 && i < buf.length) ==> buf[i] >= 0) */
    static int top = 0; 
    /*@ invariant top >= 0 && top <= buf.length */

    /*@ oti 
      DekkerLockStatic.mx == \tid ==> \old(top) == top */
    /*@ oti 
      DekkerLockStatic.mx == \tid ==> \old(buf) == buf */
    /*@ global_invariant top >= 0 && top <= buf.length */
      

    /*@ requires \tid == 1 */
    /*@ requires datum >= 0 */
    /*@ modifies top, buf */
    void put(int datum) {
	while (true) {
	    DekkerLockStatic.acquire1();
	    if (top < buf.length) {
		buf[top] = datum;
		top++;
		DekkerLockStatic.release1(); 
		break;
	    }
	    DekkerLockStatic.release1();
	}
    }


    /*@ requires \tid == 2 */
    /*@ ensures \result >= 0 */
    /*@ modifies top, buf */
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
