/**
 ** These files test the use of new instance expressions' enclosing
 ** instance pointers -- both with and without explicit enclosing
 ** instance pointers.
 **/

/*
 * First, test case where we have a non-inner class:
 */
class Top {
    // no explicit enclosing instance ptr...
    Top foo(int x) {
	return new Top();       // normal 1.0 case
    }

    int TopLevel;    // attempt to block seeing TopLevel...
    
    static class TopLevel {
	TopLevel foo(int x) {
	    return new TopLevel();  // similar, but with nested top-level class
	}

	Top2.TopLevel foo(int x, int y) {
	    return new Top2.TopLevel();  // above without inference possibility
	}
	
	TopLevel foo() {
	    return new Top.TopLevel();  // check non-simple name ok...
	}
	
	TopLevel foo(Top x) {
	    return x.new TopLevel(); // error! TopLevel isn't an inner class
	}


	// continued in New1.java...
    }
}
class Top2 {
    static class TopLevel {}
}


// continued in NewII.java...
