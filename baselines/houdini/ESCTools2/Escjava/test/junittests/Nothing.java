// Tests the parsing of \nothing in various clauses

public class Nothing {

	// These are ok.
	//@ modifies \nothing;
	//@ accessible \nothing;
	//@ subclassing_contract
	//@	callable \nothing;
	//@	accessible \nothing;

	public void n();
	// These are errors.
	//@ requires \nothing;
	//@ ensures \nothing;
	//@ diverges \nothing;
	//@ signals (Exception e) \nothing;
	//@ when \nothing;
	//@ subclassing_contract
	//@	measured_by \nothing;
	public void m();

}
