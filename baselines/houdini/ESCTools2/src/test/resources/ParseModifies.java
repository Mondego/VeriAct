public class ParseModifies {

	public boolean b,bb,bbb;

	//@ modifies b;
	public void m();

	//@ requires b;
	//@ modifies bbb;
	//@ also
	//@ requires bb;
	//@ modifies b;
	public void mm();

	public boolean r,rr,r2;
	//@ requires r;
	//@ requires r2;
	//@ {| requires bbb; modifies b; also requires b; modifies bb; |}
	//@ also requires rr;
	//@ {| requires true; modifies b; also requires false; modifies bb; |}
	public void q();

	public int i,j;
	//@ ensures j == i;
	public void z() {
		i = i+1;  //@ nowarn Modifies;
	}
}
