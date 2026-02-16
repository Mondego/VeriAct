//Tests that the helper annotation works

public class Helper {

// OK: Invariant not maintained, but is a helper routine
// A final routine may be a helper routine
	//@ ensures i == 6;
	//@ modifies i;
	//@ helper
	final public void m() {
		i = 7;
	}

// OK: Invariant not maintained, but is a helper routine
// A private routine may be a helper routine
	//@ ensures i == 6;
	//@ modifies i;
	//@ helper
	private void mm() {
		i = 7;
	}

// Error here, since the invariant is not maintained.
	//@ modifies i;
	public void p() { i=10; return; }

	public int i;

	//@ public invariant i == 0;

	public void n() {
		m();
		i=0;
	}

	public int j;

	//@ public invariant j == 100;

	//@ helper
	public Helper() {
		i = 0;
	}

	public Helper(int k) {
		this();
		j = 100;
	}
}


final class H2 {

// OK: Invariant not maintained, but is a helper routine
// A routine in a final class may be a helper routine
	//@ helper
	public void mmm() {}
}
