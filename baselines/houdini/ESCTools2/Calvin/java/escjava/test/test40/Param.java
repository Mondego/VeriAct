/* Copyright Hewlett-Packard, 2002 */

class Param {

    void put(/*@ non_null @*/ String x) {
	if ("hello".equals(x))
	    put(x);

	x = null;			// error
    }

    void doit(String y) {
	put(y);				// error
    }

    void doit2(/*@ non_null @*/ String y) {
	put(y);
    }
}
