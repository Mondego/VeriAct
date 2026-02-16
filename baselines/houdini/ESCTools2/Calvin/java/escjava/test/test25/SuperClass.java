/* Copyright Hewlett-Packard, 2002 */

class SuperClass {

    //@ invariant x!=0;
    int x;

    int y;

    SuperClass() {
	x = 1;
    }
}


class SubClass extends SuperClass {

    SubClass() { }	// no error


    // make sure we can find an invariant in a superclass:
    void foo() {
	x = 0;
    }			// error!

    void bar() {	// no error
	y = 0;
    }
}
