// Tests the desugaring of method specs
//#FLAGS: -showDesugaredSpecs -classpath .:ESCTOOLS_ROOT/specs
public class DesugaringA  extends JJ {
	boolean p,q,r,s,t;

	/*@
	requires p;
	ensures q;
	also
	requires r;
	requires t;
	ensures s;
	*/
	void m();

	/*@
		requires p;
		{|
			ensures q;
		also
			requires s;
			ensures t;
		also
			{|
				requires t;
				ensures s;
			|}
		|}
	*/
	void mm();

	/*@
		requires p;
		ensures q;
	also
		ensures r;
	*/
	void m3();

	// nothing
	void m4();

	// non_null only
	//@ non_null
	Object m5();

	Object m6(/*@ non_null */ Object o);

	/*@ exceptional_behavior
		requires p;
	also normal_behavior
		requires q;
	*/
	void m13();

	/*@
		also
		requires p;
		ensures q;
	*/
	void ppp();

	void ppp1();

	void ppp2();
}

class JJ {
	boolean pp,qq;
	void ppp();

	//@ requires pp;
	//@ ensures qq;
	void ppp1();

	
	void ppp2();
}


