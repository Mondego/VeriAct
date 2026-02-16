// Tests the \invariant_for function

public class InvariantFor {

	InvariantForA a;
	//@ ensures \invariant_for(a);
	void m() {
		a.i=0; //@ nowarn Modifies;
	}


}

class InvariantForA {
	static int i;
	//@ invariant i == 0;
}
