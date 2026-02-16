/* Copyright Hewlett-Packard, 2002 */

class Local {

    //@ requires x!=null
    void bar(String x) {
	/*@ non_null @*/ String s = x;

	//@ assert s!=null

	s = null;			// error
    }
}
