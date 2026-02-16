/**
 ** Verify that we enforce static contexts correctly in the presence
 ** of 1.1 and enclosing instance arguments.
 **/

/*
 * Test "this" in all possible contexts:
 */
class Outer {
    static void s() {
	Outer O = this;          // error: unqualified this in static context
    }

    void i() {
	Outer O = this;          // ok, non-static context
    }


    class Inner {
	Inner(Outer O) {}
    }
    class Lower extends Inner {
	Lower() {
	    super(this);         // error: constructor arguments are static
	}

	Lower(Outer O) {
	    this.super(O);       // error: enclosing instance args are static
	}
    }
}


/*
 * Ditto for "<C>.this"
 */
class Outer2 {
    int x;

    class Inner {

	static void s() {       // error because static not legal here
	    x = x;              // no error even though static context
	}

	void i() {
	    x = x;
	}

	Inner(int y) {}
    }

    class Lower extends Inner {
	Lower() {
	    super(x);           // no error even though static context
	}

	Lower(char c) {
	    Outer2.this.super(3);   // no error
	}
    }
}


/*
 * Test error messages of implicit [C.]this's.
 */
class Outer3 {
    int x;

    static void s() {
	x = 0;       // error
    }

    static void s2() {
	Outer3.x = 0;       // error
    }

    void i() {
	Outer3.x = 0;       // error
    }


    class Inner {

	void i() {
	    x = x;   // no error
	}

	void i2() {
	    Outer3.x = 0;   // error
	}

    }
}


/*
 * Also check method invocations briefly
 */
class Outer4 {
    void m() {}

    static void n() { m(); }
}
