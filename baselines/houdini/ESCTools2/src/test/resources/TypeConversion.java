// This tests that \TYPE and Class values can beb used interchangeably.


public class TypeConversion {

	//@ ghost public \TYPE t,tt;
	public Class c;

	//@ modifies t,tt,c;
	//@ ensures c == c;
	//@ ensures t <: tt;
	public void m() {
		//@ set t = \typeof(this);
		//@ set tt = \type(TypeConversion);
		c = TypeConversion.class;
	}

	//@ ensures c == t;
	public void n() {
	    c = TypeConversion.class;
	    //@ set t = \type(TypeConversion);
	}

	//@ ghost public Class cc;

	public void p() {
	    //@ set cc = \type(TypeConversion);
	    //@ set t = TypeConversion.class;
	    //@ set t = getClass();
	}

	//@ ensures \result == \typeof(this);
	//@ pure
	public Class getC() { return getClass(); } 

	/*@ ensures \result;
	    pure
	    model public boolean tt(\TYPE t) { return true; }
	*/

	//@ ensures \result;
	//@ pure
	public boolean ccc(Class c) { return true; }

	//@ pure
	public boolean ccc(String c) { return true; }

	public void s() {
		//@ assert tt(getC());
		//@ assume ccc("");
		//@ assume ccc(\type(TypeConversion));
		//-@ set t = (\TYPE)c;  // FIXME - JML does not like this
		//@ set cc = (Class)t;
	}

	public void t() {
		//+@ set cc = (\type(TypeConversion)).getClass();  // FIXME - Escjava
	}

}
