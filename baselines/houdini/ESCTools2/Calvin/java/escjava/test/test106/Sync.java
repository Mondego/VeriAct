class Sync {
    int a;
    int []b;
    
    /*@ global_invariant holder == 0 ==> a > 0 */
    /*@ oti holder == \tid ==> \old(a) == a */

    //@ requires holder != \tid
    void test1() {
	synchronized(this) {
	    // assert a > 0
	    a = a + 1;
	    // assert a > 0 
	}
    }

    void test2() {
	Sync x;

	x = new Sync();
    }
}
