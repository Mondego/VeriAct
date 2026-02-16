/**
 ** This file checks that we make sure a type cannot have the same
 ** simple name as an enclosing type.
 **/

class Dup {

    static class Dup {}            // error: repeated name

    class B {}

    static class I {
	class Dup {}        // error: repeated name

	class B {}          // siblings are ok...

	interface I {}      // error for interfaces also...
    }

    interface J {
	class J {}          // error
    }
}


/*
 * Check that supertype names do not count in this check
 */

class E extends Dup {
    class Dup {}             // ok!
}
