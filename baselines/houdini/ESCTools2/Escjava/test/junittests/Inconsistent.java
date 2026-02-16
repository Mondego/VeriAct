public class Inconsistent {
	Inconsistent();

	public void m() {
		int a,b,c,d;
		//@ assume a == b;
		//@ assume b == c;
		//@ assume a != c;
		//@ assert a == d; // Passes, but inconsistent
		//@ assert false; // does not fail
	}
	public void mm() {
		int a,b,c,d;
		//@ assume a == b;
		//@ assume b == c;
		//@ assert a == d; // Should fail
	}
	public void mmm() {
		int a,b,c,d;
		//@ assume a == b;
		//@ assume b == c;
		//@ assert a == c;  // OK and consistent
		//@ assert false; // should fail
	}
}
