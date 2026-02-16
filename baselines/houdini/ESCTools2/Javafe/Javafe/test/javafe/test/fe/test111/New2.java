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

	TopLevel bar(String[] q) {
	    return q.new TopLevel(); // error! not a reference type II!
	}


	// continued in New3.java...
    }
}
class Top2 {
    static class TopLevel {}
}
