/**
 ** This file tests the use of super constructors under 1.1 -- both
 ** with and without explicit enclosing instance pointers.
 **/

/*
 * First, test case where we have a non-inner class:
 */
class Normal extends NotInner {

    Normal() {
	// normal 1.0 case: super constructor is inferred
    }

    Normal(int x) {
	super();          // normal 1.0 case
    }

    Normal(Object x) {
	x.super();        // error!
    }
}
class NotInner {}         // ok: our constructor is inferred


/*
 * Now try case where we have an inner class enclosed in a top-level one:
 */

class LevelOne {
    class Top {}

    /*
     * Inherit from sibling inner class: (here, the super call must be
     *    passed an object of type LevelOne; such an object, namely
     *    LevelOne.this, is available)
     */
    class Bottom1 extends Top {}   // constructor is inferred successfully
    class Bottom extends Top {
	Bottom() {}         // ok: super call is inferred successfully

	Bottom(int x) {
	    super();        // ok: we infer LevelOne.this.super()
	}

	Bottom(char x) {
	    LevelOne.this.super();
	}

	Bottom(LevelOne x) {
	    x.super();
	}

	Bottom(Object x) {
	    x.super();             // error: x must be of type LevelOne
	}
    }

    /*
     * Inherit from sibling class to a top-level class: (here, the
     *    super call must be passed an object of type LevelOne; no
     *    such object is easily available so inference will fail)
     */
    static class SBottom1 extends Top {}    // error
    static class SBottom extends Top {
	SBottom() {}                 // error

	SBottom(int x) {
	    super();                 // error
	}

	SBottom(char x) {
	    LevelOne.this.super();   // error
	}

	SBottom(LevelOne x) {
	    x.super();
	}

	SBottom(Object x) {
	    x.super();               // error: x must be of type LevelOne
	}
    }
}


/*
 * Verify that the inference algorithm works correctly on multiple levels:
 */

class Inference {
    class A {
	class B {}
    }

    class C extends A.B {
	C() {}          // error: no A available
    }
    class C1 extends A.B {}  // error (same problem, different message)


    class D extends A {
	class E extends A.B {
	    E() {}              // ok: can get an A from D.this
	}
    }


    class Ambigous extends A {
	class Amb extends A {
	    class F extends A.B {}    // ok
	}
    }
}


/*
 * One more case: make sure does not consider top level nested types
 * as inner classes:
 */

class Red extends Orange.Blue {   // inherit from a top level nested type
    Red() {}                    // ok
}
class Orange {
    static class Blue {}
}
