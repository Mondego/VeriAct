// Tests the behavior of assert in java mode
//#FLAGS: -javaAssertions -sourcepath . 
public class Assert {

	//no -ea:  Unexpected exception
	// -ea: Assert error
	public void m_Exception_Assert(int i) {
		assert true : "Hi";
		assert false : 0;
	}

	// no -ea:  Unexpected exception
	// -ea:  Assert error
        //@ signals_only \nothing;
	//@ exsures (java.lang.AssertionError e) true;
	public void mm_Exception_Assert(int i) {
		assert true : "Hi";
		assert false : 0; 
	}

	// No warnings
	public void mm_Nowarn_Nowarn(int i) throws AssertionError {
		assert true : "Hi";
		assert false : 0; //@ nowarn Assert;
	}

	// No warnings
	public void mmm_Nowarn_Nowarn(int i) {
		assert true : "Hi";
	}

	// ESC Assert warning
	public void nAssertWarn(int i) {
		//@ assert true;
		//@ assert false;
	}

	// No ESC Assert warning
	public void nNowarn(int i) {
		//@ assert true;
		//@ assert false; //@ nowarn Assert;
	}

	// Postcondition warning
	//@ exsures (java.lang.AssertionError e) false;
	public void pPostwarning() throws AssertionError {
		throw new java.lang.AssertionError("E");
	}

	// Postcondition warning
        //@ signals_only RuntimeException;
	//@ exsures (java.lang.AssertionError e) true;
	public void ppPostwarning() throws AssertionError {
		throw new java.lang.AssertionError("E");
	}

	// Unexpected Exception
	// Esc/Java complains that the type in the exsures statement is not in the throws set
	// ESC/Java2 does not have this complaint.  FIXME

        //@ signals_only \nothing;
	//@ exsures (java.lang.AssertionError e) true;
	public void pExceptionwarningShouldCompilerError()  {
		throw new java.lang.AssertionError("E");
	}


	// No warning
	public void qNoWarning(int i) throws AssertionError {
		throw new java.lang.AssertionError("E");
	}

	// ESC Unexpected exception warning
	public void qqExceptionWarning(int i) {
		throw new java.lang.AssertionError("E");
	}

	// No warning
	public void rNowarning() throws AssertionError {
		throw new AssertionError();
	}
}
