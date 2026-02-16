// Testing parsing of \not_specified

public class NotSpecified {

	//@ requires \not_specified;
	//@ modifies \not_specified;
	//@ ensures \not_specified;
	//@ signals (Exception e) \not_specified;
	//@ diverges \not_specified;
	//@ when \not_specified;
	//@ duration \not_specified;
	//@ working_space \not_specified;
        //@ also code_contract
	//@	measured_by \not_specified;
	//@     accessible \not_specified;
	//@	callable \not_specified;
	public void m() throws Exception;
}
