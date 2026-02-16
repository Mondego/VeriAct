/* Copyright Hewlett-Packard, 2002 */

/**
 ** Verify that we do null checks for the enclosing instance arguments
 ** of new and super.
 **/

class Outer {
    class Inner {
    }

    void test1(Outer O) {
	Inner I = O.new Inner();        // null error
    }

    class Lower extends Inner {
	Lower(Outer O) {
	    O.super();                  // null error
	}
    }
}
