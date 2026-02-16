// This tests that 'this' and the formals get mapped when 
// setting the modified variables to arbitrary results

public class Unch {

	public int[] a = new int[10];
	//@ public invariant a != null && a.length >= 10;

	//@ modifies a[i];
	public void m(int i);

	public void mm() {
		m(1);
		//@ assert \not_modified(a[0]);
	}

	public void mm1() {
		m(1);
		//@ assert \not_modified(a[1]); // FAILS
	}

		
	public int n;
	Unch x = new Unch();

	//@ modifies n;
	public void mn();

	//@ requires x != this && x != null;
	public void mnn() {
		x.mn();
		//@ assert \not_modified(this.n);
	}

	//@ requires x != this && x != null;
	public void mnn1() {
		x.mn();
		//@ assert \not_modified(x.n); // FAILS
	}
}
