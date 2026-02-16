// Tests for error messages when there are modifies clauses on pure routines

public class PureModifies extends PureModifiesS {

	//@ requires true;
	void notpure();

	//@ pure
	void m1();

	//@ modifies \nothing;
	//@ pure
	void m2();

	//@ modifies \everything; // ERROR
	//@ pure
	void m3();

	//@ requires true;
	//@ pure
	void m4();

	//@ behavior
	//@ requires true;
	//@ pure
	void m5();

	//@ behavior
	//@ modifies \nothing;
	//@ pure
	void m6();

	//@ behavior
	//@ modifies \everything; // ERROR
	//@ pure
	void m7();


	//@ requires true;
	//@ {|
	//@	modifies \nothing;
	//@	also
	//@	modifies \everything; // ERROR
	//@ |}
	//@ pure
	void m8();


/*@
	modifies \everything;  // ERROR
	pure model void mm();

	requires true;
	pure model void mm1();

	modifies \everything;  // ERROR
	pure model PureModifies(Object o);

	requires true;
	pure model PureModifies(Object o, Object oo);
*/


	//@ modifies \everything; // ERROR
	//@ pure
	PureModifies();

	//@ requires true;
	//@ pure
	PureModifies(int i, int j);

	// modifies this.*; // OK
	//@ pure
	PureModifies(int i);


	//@ also modifies \everything;  // ERROR
	void ms() {}
}

//@ pure
class PureModifiesA {

	//@ modifies \everything;  // ERROR
	PureModifiesA();
}

class PureModifiesS {

	//@ pure
	void ms() {}
}
