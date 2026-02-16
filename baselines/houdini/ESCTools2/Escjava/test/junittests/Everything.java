public class Everything extends S {

	//@ modifies \everything;
	//@ modifies \nothing;
	//@ modifies \not_specified;
	public void m() {}

	//@ also modifies \everything;
	public void n() {}
}

class S {
	public int i;

	//@ modifies i;
	public void n() {}
}
