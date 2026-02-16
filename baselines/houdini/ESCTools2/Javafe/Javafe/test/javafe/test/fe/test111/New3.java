/**
 ** These files test the use of new instance expressions' enclosing
 ** instance pointers -- both with and without explicit enclosing
 ** instance pointers.
 **/


class Top {
    // no explicit enclosing instance ptr...
    Top foo(int x) {
	return new Top();       // normal 1.0 case
    }

    int TopLevel;    // attempt to block seeing TopLevel...
    
    static class TopLevel {
	// ...

	TopLevel foo(Object x) {
	    return x.new TopLevel(); // error! x has no TopLevel type member
	}


	// continued in New4.java...
    }
}
class Top2 {
    static class TopLevel {}
}
