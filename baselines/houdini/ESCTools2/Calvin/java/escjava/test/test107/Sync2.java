/* Copyright Hewlett-Packard, 2002 */

class Sync2 {
    /*@ writable_if holder == \tid */ int a;
    
    /*@ global_invariant holder == 0 ==> a > 0 */    

    //@ requires holder != \tid
    void test1() {
	synchronized(this) {
	    a = a + 1;
	}
	a = 1;
    }

}


