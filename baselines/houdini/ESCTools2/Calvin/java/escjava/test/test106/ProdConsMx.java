
/** ProdCons is a producer-consumer class that
 ** implements a producer-consumer system. The
 ** central data structure is a stack, represented
 ** by a top, and an array "buf". Access to these
 ** elements is protected using a ghost mutex variable.
 ** 
 ** There are two threads in the system. Thread 1
 ** is a producer thread, and executes the "put()" 
 ** method. Thread 2 is a consumer thread and executes
 ** the "get()" method.
 **/
class ProdConsMx {
    static final int BUF_SIZE = 10;
    /*@ non_null */ int[] buf = 
	new int[BUF_SIZE];
    int top = 0; /*@ invariant 
			  top >= 0 && top <= buf.length */
  

    //@ ghost public int mx;

    /*** These ghost variables, OTIs and global 
	 invariants are same as in the DekkerLock class
    ***/
    /*@ global_invariant mx == 0 || mx == 1 || mx == 2 */
    /*@ oti \tid == \old(mx) ==> \old(mx) == mx */
    /*@ oti \tid == \old(mx) ==> \tid == mx */
    /*@ oti \tid != \old(mx) ==> \tid != mx */
    /*---------- end mx invariants ---------*/    


    /*@ oti \old(mx) == \tid ==> \old(top) == top */
    /*@ oti \old(mx) == \tid ==> \old(buf) == buf */
    /*@ global_invariant top >= 0 && top <= buf.length */
    /* global_invariant top >= 0 && top <= 10 */
    /* oti \old(BUF_SIZE) == BUF_SIZE */
      

    /*@ requires \tid == 1 */
    /* NOT CHECKING THIS FOR NOW -- 
      requires datum >= 0 */
    /*@ modifies top, buf, mx */
    void put(int datum) {
	while (true) {
	    //@ assume mx == 0;
	    //@ set mx = \tid;
	    if (top < buf.length) {
		buf[top] = datum;
		//@ assert mx == \tid;
		int tmptop = top;
		//@ assert mx == \tid;
		top = tmptop + 1;
		//@ assert mx == \tid;
		//		top++;
		//@ set mx = 0;
		/* might be interesting
		   to leave this out and 
		   see what the checker
		   turns up */
		break;
	    }
	    //@ set mx = 0;
	}
    }


    /*@ requires \tid == 2 */
    /* NOT CHECKING THIS FOR NOW -- 
       ensures \result >= 0 */
    /*@ modifies top, buf, mx */
    int get() {
	int retVal = -1;
	while (true) {
	    //@ assume mx == 0;
	    //@ set mx = \tid;
	    if (top > 0) {
		retVal = buf[top-1];
		top--;
		//@ set mx = 0;
		break;
	    }
	    //@ set mx = 0;
	}
	return retVal;
    }


}
