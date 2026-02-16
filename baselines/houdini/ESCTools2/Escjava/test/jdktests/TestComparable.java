// $Id$

public class TestComparable extends LocalTestCase {

    private final /*@non_null*/ Comparable c = new Integer(331);

    public void testNull() {
	// Test illegal null argument
	try {
	    c.compareTo(null); // --> warn NonNull
	} catch (Exception e) {
	    // ignore
	}
	//@ assert false; // TEST FOR CONSISTENCY
    }

    public void testIncomparableArgument() {
	try {
	    Object o = new Object(); // o is not comparable.
	    c.compareTo(o);
	    //@ unreachable;
	} catch (Exception e) {
	    //@ assert e instanceof ClassCastException;
	}
	//@ assert false; // TEST FOR CONSISTENCY
    }

    public void testReflexivity() {
	int i = c.compareTo(c);
	//@ assert i == 0;
	//@ assert false; // TEST FOR CONSISTENCY
    }

    public void testSymmetry() {
	// Note that Integer's are Comparable
	Comparable cc = new Integer(1);

	int i = c.compareTo(cc);
	int j = cc.compareTo(c);
	//@ assert java.lang.Comparable.sgn(i) == -java.lang.Comparable.sgn(j);
	//XXX assert i == -j; // this cannot be deduced from the spec.
	//@ assert false; // TEST FOR CONSISTENCY
    }
}
