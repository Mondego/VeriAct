/* Copyright Hewlett-Packard, 2002 */

import java.lang.Math;

/** ProdCons is a producer-consumer class that
 ** implements a producer-consumer system. The
 ** central data structure is a stack, represented
 ** by a top and an array "buf". Access to these
 ** elements is protected using a so-called "Dekker"
 ** lock, implemented by the DekkerLock class.
 ** 
 ** There are two threads in the system. Thread 1
 ** is a producer thread, and executes the "put()" 
 ** method. Thread 2 is a consumer thread and executes
 ** the "get()" method.
 **/
class ProdCons {

    final int BUF_SIZE = 10;
    /*@ non_null */ int[] buf;
    /*@ global_invariant (\forall int i; 
      (i >= 0 && i < buf.length) ==> buf[i] >= 0) */
    int top; /*@ global_invariant 
	       top >= 0 && top <= buf.length */
    /*@ non_null */ final DekkerLock lock;

    /*@ oti 
      lock.mx == \tid ==> \old(top) == top */
    /*@ oti 
      lock.mx == \tid ==> \old(buf) == buf */

    /* oti
      (lock.mx == \tid) == (\old(lock.mx) == \tid) */

      
    ProdCons() {
	top = 0;
	buf = new int[BUF_SIZE];
	lock = new DekkerLock();
    }


    /*@ requires \tid == 1 */
    /*@ requires datum >= 0 */
    /*@ modifies top, buf */
    void put(int datum) {
	while (true) {
	    lock.acquire1();
	    if (top < buf.length) {
		buf[top] = datum;
		top++;
		lock.release1(); 
		break;
	    }
	    lock.release1();
	}
    }


    /*@ requires \tid == 2 */
    /*@ modifies top, buf */
    /*@ ensures \result >= 0 */
    int get() {
	int retVal = -1;
	while (true) {
	    lock.acquire2();
	    if (top > 0) {
		retVal = buf[top-1];
		top--;
		lock.release2();
		break;
	    }
	    lock.release2();
	}
	return retVal;
    }


}
