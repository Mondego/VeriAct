/* Copyright Hewlett-Packard, 2002 */

/** ProdCons implements a producer-consumer system. The
 ** central data structure is a stack, represented
 ** by a top, bottom(0, in this case), and a buffer. 
 ** Access to these elements is protected using Java 
 ** synchronization.
 ** 
 ** There are two methods: get() which gets an
 ** entry from the shared buffer and is executed
 ** by the consumer, and put() which puts in a
 ** new entry and is executed by the producer thread(s). 
 ** Both methods only return on performing the action.
 **  
 ** The shared buffer satisfies the object invariant
 ** that every element of the buffer is non-negative.
 **
 **/
class ProdConsSync {
  static final int BUF_SIZE = 10;
  /*@ monitored */ /*@ non_null */ int[] buf = 
	new int[BUF_SIZE]; 
    /*@ invariant (\forall int i; 
      (i >= 0 && i < buf.length) ==> buf[i] >= 0) */    
  /*@ monitored */ int top = 0; 
  /*@ invariant top >= 0 && top <= buf.length */
  
    /*@ requires datum >= 0 */
    /*@ modifies top, buf, this.holder */
    /* performs (top, buf) 
      { (top == \old(top) + 1) && 
        (buf[\old(top)] == datum) } */ 
    void put(int datum) {
	while(true) {
	    synchronized(this) {
		if (top < buf.length) {
		    buf[top] = datum;
		    top++;
		    return;
		}
	    }
	}
    }


    /*@ ensures \result >= 0 */
    /*@ modifies top, buf */
    int get() {
      int retVal = -1;
      while(true) {
	  synchronized(this) {
	      if (top > 0) {
		  retVal = buf[top-1];
		  top--;
		  return retVal;
	      }
	  }
      }
    }

    static void dumbo() {

    }

}
