/**
 ** Test escjava's reasoning about member inner classes, part VI
 **/

import java.io.IOException;


/*
 * Verify that exsures clauses on enclosing instance variables work properly.
 */
class Outer {
    int x;

    class Inner {
	//@ modifies x
	//@ ensures x<5
	//@ exsures (IOException E) x>10
	void m() throws IOException { x = 3; }
    }

    void test() {
	Inner I = new Inner();
	x = 6;
	try {
	    I.m();
	    //@ assert x<5
	} catch (IOException E) {
	    //@ assert x>10
	}
    }
}
