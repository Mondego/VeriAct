/* Copyright Hewlett-Packard, 2002 */

class C {

    void foo() {

	Object a = new Object();
	for(int i=0; i<10; i++) {
	    //@ assert i>=0
	    a = a;
	    //@ assert a != null
	}
    }
}
