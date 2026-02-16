// This tests modifies checks with side-effects on the LHS.
public class ModChecks4 {
	int x,y;
	//@ non_null
	int[] a = new int[4];
	//@ invariant a.length > 2; 

	//@ requires x == 0;
	//@ modifies x, a[*];
	void m() {

		a[x++] = 0;
		//@ assert x == 1;
	}

	//@ requires y == 0;
	//@ modifies x,y;
	void mm() {
		f(y++).x = 0;
		//@ assert y == 1;
	}

	//@ modifies \nothing;
	//@ ensures \result == this;
	ModChecks4 f(int i) {
		return this;
	}
}
