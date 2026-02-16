/* Copyright Hewlett-Packard, 2002 */

class C {

    void foo() {

	for(int i=0; i<10; i++) {
	    //@ assert i>=0
	}
    }
}
