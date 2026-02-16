// Testing initially clauses

public class Initially {


	public int i = 5;

	//@ initially i == 6;

	//@ modifies i;
	void m() {} // OK

	//@ modifies this.*;
	public Initially() {
		i = 6;
	} // OK

	//@ modifies this.*;
	public Initially(int j) {
	} // ERROR
}

class InitiallyA {

	public int j;
	//@ modifies j;
	InitiallyA();

	//@ initially j == 4;

	//@  ensures j == 14;
	public void p();
}

class InitiallyB {

	public InitiallyA a;
	InitiallyB();

	void m() {
		a = new InitiallyA();			//@ nowarn Modifies;
		//@ assert a.j == 4; // OK
		a.p();			//@ nowarn Modifies;
		//@ assert a.j != 14;
	}

	void mm() {
		a = new InitiallyA();			//@ nowarn Modifies;
		//@ assert a.j != 4;  // ERROR
	}

}

class InitiallyC {

	static public int j;
	//@ modifies j;
	InitiallyC();

	//@ initially j == 4;
}

class InitiallyD {

	static public InitiallyC a;
	InitiallyD();

	void m() {
		a = new InitiallyC();			//@ nowarn Modifies;
		//@ assert a.j == 4; // OK
	}

	void mm() {
		a = new InitiallyC();			//@ nowarn Modifies;
		//@ assert a.j != 4;  // ERROR
	}

}
		
// Test the specs of constructors

class InitiallyE {

	//@ initially i == 19;
	public int i;

	InitiallyE();

	InitiallyE(int i);
}

class InitiallyF extends InitiallyE {

	//@ initially i == 18;

	public InitiallyF() {
		// implied super();
		//@ assert i == 19;
		i = 18;			//@ nowarn Modifies;
	}

	public InitiallyF(float f) {
		// implied super();
		//@ assert i == 20; // ERROR
		i = 18;			//@ nowarn Modifies;
	}

	public InitiallyF(int j) {
		super(j);
		//@ assert i == 19;
		i = 18;			//@ nowarn Modifies;
	}

	public InitiallyF(int j, int k) {
		super(j);
		//@ assert i == 20; // ERROR
		i = 18;			//@ nowarn Modifies;
	}

	public int j;

	//@ ensures j == 30;
	//@ helper
	public InitiallyF(int k, int z, int m) {
		j = 30; 			//@ nowarn Modifies;
	} // would be error if the constructor were not a helper

	public InitiallyF(Object o) {
		this(0,0,0);
		//@ assert i == 18; // ERROR -- constructor called is helper
		i = 18;			//@ nowarn Modifies;
	}
}
