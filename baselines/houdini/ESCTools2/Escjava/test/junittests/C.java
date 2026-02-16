//#FLAGS: -quiet
// This tests a simple case that caused a null pointer exception.
public class C {

	C() throws Exception {
		helperMethod();
		throw new Exception("E");
	}

	//@ helper
	private void helperMethod() {}
}
