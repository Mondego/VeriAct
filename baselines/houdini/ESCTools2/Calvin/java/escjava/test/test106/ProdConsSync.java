import java.lang.Math;

/** ProdCons is a producer-consumer class that
 ** implements a producer-consumer system. The
 ** central data structure is a stack, represented
 ** by a top and an array "buf". Access to these
 ** elements is protected using Java synchronization.
 ** 
 ** There are two methods: get() which gets an
 ** entry from the shared buffer and is executed
 ** by the consumer, and put() which puts in a new 
 ** entry and is executed by the producer thread(s). 
 **  
 **
 **/
class ProdConsSync {
  static final int BUF_SIZE = 10;
  /*@ monitored */ /*@ non_null */ int[] buf = 
	new int[BUF_SIZE];
  /*@ monitored */ int top = 0; 
  /*@ invariant top >= 0 && top <= buf.length */
  
    /*@ requires \tid == 1 */
    /* NOT CHECKING THIS FOR NOW -- 
      requires datum >= 0 */
    /*@ modifies top, buf */
    void put(int datum) {
      synchronized(this) {
	  if (top < buf.length) {
	    buf[top] = datum;
	    top++;
	  }
	}
    }


    /*@ requires \tid == 2 */
    /* NOT CHECKING THIS FOR NOW -- 
       ensures \result >= 0 */
    /*@ modifies top, buf */
    int get() {
      int retVal = -1;
      synchronized(this) {
	if (top > 0) {
	  retVal = buf[top-1];
	  top--;
	}
      }
      return retVal;
    }


}
