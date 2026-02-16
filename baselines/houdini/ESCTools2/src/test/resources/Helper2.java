// Tests errors in using helper

public class Helper2 {

	//@ helper
	public void m(); // ERROR

	//@ helper
	protected void m2(); // ERROR

	//@ helper
	void m3(); // ERROR
}
