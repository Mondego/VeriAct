public class UnchangedPrecondition {
	public boolean p,q;
	public int i,j;

    /*@ requires p;
	modifies i;
      also
	requires q;
	modifies j;
    */
    public void m();

    //@ requires p && !q;
    public void mm() {
	m();
	//@ assert \not_modified(j);
    }

    //@ requires p && !q;
    public void mm1() {
	m();
	//@ assert \not_modified(i); // FAILS
    }

    //@ requires q;
    public void mma() {
	if (!p) {
		m();
		//@ assert \not_modified(i); // OK
	}
    }


}
