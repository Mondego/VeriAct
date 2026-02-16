public class NotModified {

	public int i,j,k;

	//@ modifies i,j;
	//@ ensures \not_modified(i,j);
	public void m() {
		++i;
	} // ERROR - i

	//@ modifies i,j;
	//@ ensures \not_modified(i,j);
	public void m2() {
		++i;
		++j;
	} // ERROR - i,j

	//@ modifies i,j,k;
	//@ ensures \not_modified(k-j);
	public void m3() {
		++k;
		++j;
	} // OK

	//@ modifies i,j,k;
	//@ ensures \not_modified(i,j,k);
	public void m4() {
		++k;
		++j;
	} // ERROR - j,k

	//@ modifies i,j;
	//@ ensures \not_modified(i,j);
	public void mm() {
	} // OK

	//@ modifies i;
	//@ ensures \not_modified(i);
	public void mmm() {
	} // OK

	//@ modifies i;
	//@ ensures \not_modified(\old(i));
	public void mold() { ++i; }
}
