// Tests that a datagroup is expanded when a called method's specs are added in
// to the VCs
public class ModifiesUse  {
	ModifiesUse();
	//@ model public JMLDataGroup list;

	public int j; //@ in list;

	//@ modifies j;
	public void m();

	public void mm() {
		j = 25;
		m();
		//@ assert j == 25; // ERROR
	}

	//@ modifies list;
	public void mx();

	public void mmx() {
		j = 25;
		mx();
		//@ assert j == 25; // ERROR - should be
	}


	//@ modifies \nothing;
	public void mn();

	//@ modifies \everything;
	public void me();

	int kk;
	//@ non_null
	int[] a = new int[10];
	//@ invariant a.length > 5;

	Object o;

	//@ modifies a, a[0], a[*], o.*, a[2..3];
	public void ma();

	//@ modifies a, a[0], a[*], o.*, a[2..3];
	//@ modifies \everything;
	void mmm() {
		mn();
		me();
		ma();
	}
}

	
