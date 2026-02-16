public class Everything2 {

	public int i;

	public int j;

	//@ modifies i,j;
	//@ modifies i, \nothing;
	//@ modifies \everything, i;
	//@ modifies i, \not_specified;
	public void m() {}
}
