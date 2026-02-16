public class ModelVarState {

	int sz; //@ in size;
	//@ model public int size;
	//@ represents size = sz +1;

	//@ requires sz == 1;
	//@ modifies size;
	//@ ensures size == \old(size)*2;
	void m() {
		//@ assert size == size;
		//@ assert size == 2;
		sz = 2;
		//@ assert size == 3;
		sz = 3;
		//@ assert size == 4;
	}

	//@ modifies \everything;
	//@ also
	//@ requires size == 2;
	//@ ensures size == \old(size)*2; // ERROR
	void mm() {
		sz = 2;
	}

	//@ ensures size == 3;  // OK
	void mmm() {
		sz = 2;  // OK - modifies everything
	}

	//@ modifies \nothing;
	void mmmm() {
		sz = 2;  // ERROR - no modifies
	}


	//@ public model int norep;
	//@ modifies norep;
	//@ ensures norep == norep;
	void m5() {}

}
