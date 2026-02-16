/**
 ** These files test the use of new instance expressions' enclosing
 ** instance pointers -- both with and without explicit enclosing
 ** instance pointers.
 **/


/*
 * Now try case where we have an inner class enclosed in a top-level one:
 */

class LevelOne {
    class Inner {}    // the target inner class

    class EnclosingAvailable {
	// inference will work here

	Inner foo(int x) {
	    return new Inner(); // ok: we infer LevelOne.this.new Inner()
	}

	Inner foo(char x) {
	    return LevelOne.this.new Inner();
	}

	Inner foo(LevelOne x) {
	    return x.new Inner();
	}
    }
    Inner available = new Inner();    // ok, infer this.new Inner()

    static class EnclosingUnavailable {
	// inference will not work here

	Inner foo(int x) {
	    return new Inner();                 // error
	}

	Inner foo(char x) {
	    return LevelOne.this.new Inner();   // error
	}

	Inner foo(LevelOne x) {
	    return x.new Inner();
	}
    }
    static Inner unavailable = new Inner();    // OK
}


/*
 * Verify that the inference algorithm works correctly on multiple levels:
 */

class Inference {
    class A {
	class B {}
    }

    class C {
	A.B foo() {
	    return new A.B();          // OK
	}
    }


    class D extends A {
	class E {
	    A.B foo() {
		return new A.B();      // ok: can get an A from D.this
	    }
	}
    }


    class Ambigous extends A {
	class Amb extends A {
	    class F {
		A.B foo() {
		    return new A.B();      // ok
		}
	    }
	}
    }
}
