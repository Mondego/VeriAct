/**
 ** This file tests that we check that inner classes members may not
 ** be static, part I
 **/

// Static members in outside classes are ok:
class OutsideClass { 
    static { }

    static final int y=3;
    static int x;

    static void m() {};

    static class S {};

    interface I {};
}


// Ditto for top-level classes:
class OutsideClass2 { 
    static class TopLevel {
	static { }

	static final int y=3;
	static int x;

	static void m() {};

	static class S {};

	interface I {};
    }
}

// But, not in inner classes:
class OutsideClass3 { 
    class InnerClass {
	static { }                     // error

	static final int y=3;          // *not* an error!
	static int x;                  // error

	static void m() {};            // error

	static class S {};             // error

	interface I {};                // error (implicitly static)
    }
}


// test case for nested interfaces as well:
class OutsideClass4 { 
    interface Nested {
	static final int y=3;
	static int x = 0;

	static void m();               // error: interface methods never static

	static class S {};

	interface I {};

	// continued in TestStaticMembers2.java
    }
}




// Note: since interfaces are implicitly static, there is no such
// thing as a (legal) inner interface.
