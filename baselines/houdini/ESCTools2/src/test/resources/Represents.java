public class Represents {

// ALso need: model fields from super class, model fields from another class
// Model fields used in pre, post, assert, assume, invariant
// Model fields not allowed as lhs of set, how about in rhs? FIXME

	/*@ spec_public */ private int length = 0;

	static public int a = 0;
	static public int b;
	//@ private ghost boolean bbb;
	//@ ghost protected boolean bbbb;
	//@ ghost public boolean bb;

	//@ public invariant bb;

	//@ ensures bb;
	public Represents() {
		//@ set bb = true;  //@ nowarn Modifies;
	}

	//@ public model int size;
	//@ private represents size <- length+1;
	//@ axiom a == b;

	//@ requires size > 0;
	public void m() {}

	//@ modifies length, size;
	//@ ensures length == 1;
	//@ ensures size > 2;
	public void n() {
		length = 1;
		return;
	} // ERROR - postcondition not established

	//@ modifies length, size;
	//@ ensures size > 1;
	public void nn() {
		length = 1;
		return;
	} // OK

	//@ requires size > 1 && bb;
	public void p() {}
}
