// Tests the parsing of \everything in various clauses.

public class EverythingParse {
	// These are ok.
	//@ modifies \everything;
	//@ modifies \everything if true;  // deprecated
	//@ accessible \everything;
	//@ accessible \everything if true;
	//@ subclassing_contract
	//@	callable \everything;
	//@	accessible \everything;
	public void n();

	// These are errors.
	//@ requires \everything;
	//@ ensures \everything;
	//@ diverges \everything;
	//@ signals (Exception e) \everything;
	//@ when \everything;
	//@ subclassing_contract
	//@	measured_by \everything;
	public void m();

}
