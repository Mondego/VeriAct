// Other miscellaneous tests of checking modifies clauses

public class ModChecks7 {

	public ModChecks7();

	//@ non_null
	public ModChecks7 o = new ModChecks7();
	//@ non_null
	public ModChecks7 oo = new ModChecks7();

	public int i;

	// Tests that mappding of 'this' works ok

	//@ modifies o.i;
	public void mi() {
		o.pi(); // OK
	}

	//@ modifies i;
	public void mi2() {
		o.pi(); // ERROR
	}

	//@ modifies o.i;
	public void mi3() {
		pi(); // ERROR
	}

	//@ modifies i;
	public void pi();

	// Tests old and new states

	//@ modifies o.i;
	public void mo1() {
		ModChecks7 ooo = o;
		ooo.i = 0; // OK
	}

	//@ modifies o.i;
	public void mo2() {
		o = oo;
		o.i = 0; // ERROR
	}

	//@ modifies oo.i;
	public void mo3() {
		o = oo;
		o.i = 0; // OK
	}


		
		
}
