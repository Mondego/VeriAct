// Tests recovery from error conditions in parsing sequences of modifiers
//#FLAGS: -nocheck 
public class ModParsingErrors  extends ModParsingJJ {
	boolean p,q,r,s,t;

	/*@
	requires  p;
	{|
	|} // ERROR -- empty section
	*/
	void m7();

	/*@
	requires p;
	|}		// ERROR - no opening {|
	*/
	void m7a();

	/*@ requires p;
	pure
	non_null     
	ensures q;   // ERROR - out of order -- FIXME
	*/
	Object m8();

	/*@ behavior 
		requires p;
	also
		requires q;
		normal_behavior    // ERROR -- out of order
		ensures r;
	*/
	void m9();

	/*@ behavior
		requires p;
	implies_that
		normal_example     // ERROR - example in implies section
			requires q;
	for_example
		exceptional_behavior   // ERROR - behavior in examples section
			requires r;
	*/
	void m10();

	/*@ also behavior        // ERROR -- no also
		requires r;
	also
		requires p;
		{|
			behavior	// ERROR - misplaced
			requires q;
		|}
	*/
	void m11();

	/*@ behavior
		requires p;
		{|
			requires q;
		also
		|}		// ERROR -- empty sequence
	|}			// ERROR -- no matching |}
	*/
	void m12();

	/*@ requires true;
	also
	also			// ERROR - empty sequence
		requires p;
	*/
	void m12a();

	/*@ behavior
		requires p;
		{|
			ensures q; 
		pure		// ERROR - expected closing |}
		|}
	*/
	void m13();

	/*@ normal_behavior
		requires p;
		signals(Exception) true; // ERROR -- no signals in normal section
	also exceptional_behavior
		requires q;
		ensures true;     // ERROR -- no ensures in exceptional section
	*/
	void m14();

	/*@
		requires p;
	*/
	void ppp();  // ERROR - needs an also


	/*@ for_example
		requires p;
		ensures q;
	also
		model_program { } // ERROR - no model program in examples
	*/
	void m15();

	/*@ requires true;
	{|
		ensures false;
	also
		model_program {} // ERROR - no nested model program
	|}    
	*/
	void m16();

	/*@
		requires true;
	also
		model_program {} // OK
	also
		requires true;
		model_program{} // ERROR
	also
		model_program {} // ERROR
		ensures true;
	*/
	void m17();
}

class ModParsingJJ {
	void ppp();
}
