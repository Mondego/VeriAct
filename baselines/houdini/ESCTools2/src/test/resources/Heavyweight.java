public class Heavyweight {

	public boolean b,bb,bbb;

	//@ public normal_behavior
	//@	requires true;
	//@	ensures b;
	//@ also
	//@ public behavior
	//@	requires bb;
	//@	signals (Exception) !b;
	//@ also
	//@ public normal_behavior
	//@	requires bbb;
	//@	{| 
	//@		ensures b;
	//@	also
	//@		diverges false;
	//@	|}
	public void m() throws Exception;

	//@	requires false;
	//@	ensures false;
	//@ also
	//@	ensures b;
	//@	signals (java.lang.ClassCastException e) true;
	public void n() throws ClassCastException;


	//@ requires b;
	//@ modifies b,bb;
	//@ also
	//@ requires bb;
	//@ modifies \nothing;
	public void q();

	//@ requires true;
	//@ {| ensures false;
	//@  also
	//@    diverges false; // ERROR - missing closing |}
	public void badEnd();

	//@ requires b;
	//@ also		// ERROR - dangling also
	public void danglingAlso();

	/*@ requires true;
	      ensures true;
            also
              diverges false;
            |}			// ERROR - no opening {|
         */
         public void missingBegin();

	//@ ensures true;
	public Heavyweight() {}
}

