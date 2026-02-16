// Tests reasoning with constraints

public class Constraint {

	public int i;

	//@ constraint i == \old(i);

	//@ modifies i;
	public void m() {} // OK

	//@ modifies i;
	public void mm() {
		++i;
	} // ERROR

	//@ modifies i, this.*;
	public Constraint() {} // OK
}


class ConstraintA {
	ConstraintA();

	//@ non_null
	ConstraintB o = new ConstraintB();

	//@ modifies o.j;
	void m() {
		o.j = 5;
		o.m();	
		//@ assert o.j == 6; // OK
	}

	//@ modifies o.j;
	void mm() {
		o.j = 5;
		o.m();	
		//@ assert o.j != 6; // ERROR
	}

}

class ConstraintB {

	//@ constraint j == \old(j) + 1;

	//@ modifies j;
	void m();

	public int j;
}


class ConstraintSA {
	ConstraintSA();

	//@ non_null
	static ConstraintC o = new ConstraintC();

	//@ modifies o.sj;
	static void m() {
		o.sj = 5;
		o.sm();	
		//@ assert o.sj == 6; // OK
	}

	//@ modifies o.sj;
	static void mm() {
		o.sj = 5;
		o.sm();	
		//@ assert o.sj != 6; // ERROR
	}

}
class ConstraintC {
	//@ constraint sj == \old(sj) + 1;

	public static int sj;

	//@ modifies sj;
	static void sm();

	//@ modifies sj;
	//@ helper
	static void sh() {}

	//@ modifies sj;
	static void sz() {
		sj = 0;
		sm();
		//@ assert sj == 1;
		sh();
		//@ assert sj == 2; // ERROR - sh is helper
	}
	
}

class ConstraintR implements ConstraintI {

	public int k;
	//@ constraint k == \old(k) + 3;

}

interface ConstraintI {

	//@ constraint !(this instanceof ConstraintX);
}


class ConstraintX extends ConstraintR implements ConstraintI {
	//@ modifies k;
	public void m() {
		k += 3;
	} // ERROR - fails ConstraintI
	//@ modifies k;
	public void mm() {
		k += 4;
	} // ERROR - fails ConstraintR
}
