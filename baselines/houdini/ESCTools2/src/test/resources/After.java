// Tests that specifications that are after the declaration are read
//#FLAGS: -checkPurity
public class After {

	native public boolean m()
		//@ requires true;
		//@ ensures false;
	;

	native public boolean purem()
		//@ requires true;
		//@ ensures false;
		//@ pure
	;

	public int n()
		/*@ requires true;
		    ensures false;
		    ensures purem() && m();
		*/
		//@ pure
	{
	}
}
