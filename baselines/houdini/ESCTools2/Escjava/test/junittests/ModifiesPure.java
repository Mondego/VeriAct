// Tests for warnings when pure methods have modifies clauses.
//@ pure
public class ModifiesPure {

	public int i;

	//@ modifies i;			// WARNING
	public void m();

	//@ public normal_behavior
        //@ {|
	//@	assignable i;		// WARNING
	//@ also
	//@	assignable \nothing;    // OK
        //@ |}
	public void n();
}


class ModifiesPureA {

	public int i;

	//@ modifies i;			// WARNING
	//@ pure
	public void m();

	//@ public normal_behavior
        //@ {|
	//@	assignable i;		// WARNING
	//@ also
	//@	assignable \nothing;  // OK
        //@ |}
	public /*@ pure @*/ void n();
}
