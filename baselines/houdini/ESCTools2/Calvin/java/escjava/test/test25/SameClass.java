/* Copyright Hewlett-Packard, 2002 */

class SameClass {

    //@ invariant x!=0;
    int x;

    int y;


    // This test makes sure we know that there is an implicit reference
    // to x in the constructor:
    SameClass() { }	// error!


    void foo() {
	x = 0;
    }			// error!

    void bar() {	// no error
	y = 0;
    }
}
