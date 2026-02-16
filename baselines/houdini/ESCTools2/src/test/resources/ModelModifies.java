// Tests that model variables in preconditions that become modifies 
// conditionals are translated correctly

public class ModelModifies {

	public int s; //@ in sm;

	//@ model public int sm;

	//@ requires s == 2;
	//@ modifies s;
	public void m() {
	    s = 2;
	}

	//@ requires sm == 2;
	//@ modifies sm;
	public void mm() {
	    s = 2;
	}

	//@ ensures \result == i+4;
	//@ pure
	public int p(int i) {
	    return i+4;
        }

	//@ requires sm == p(0);
	//@ modifies s;
	public void  mmm() {
		s = 2;
		s = 3;
	}
}
