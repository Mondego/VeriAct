// Tests the logic of nonNull on the result of a method, including inheritance


public class NonNullResult extends NonNullResultS {

	public NonNullResult();

	//@ pure
	public NonNullResultS m1(int i) {
		return null;
	} // OK

	//@ pure non_null
	public NonNullResultS m2(int i) {
		return null;
	} // FAILS

	//@ pure
	public NonNullResultS m3(int i) {
		return null; 
	} // FAILS - super class spec

	//@ pure non_null
	public NonNullResultS m4(int i) {
		return null;
	} // FAILS

	//@ also requires i <= 0 && i >=-10;
	//@ pure
	public NonNullResultS m5(int i) {
		return null; 
	} // FAILS - super class spec

	public void mm() {
		/*@ non_null */ NonNullResultS o = new NonNullResultS();
		o = m1(0); // FAILS
		o = m2(0); // OK
		o = m3(0); // OK
		o = m4(0); // OK
		o = super.m1(0); // FAILS
		o = super.m2(0); // FAILS
		o = super.m3(0); // OK
		o = super.m4(0); // OK
		o = m5(0);
		o = m5(-20); // FAILS - Precondition
		o = m5(-1); // FAILS
		o = m5(1);
	}

}

class NonNullResultS {

	//@ pure
	public NonNullResultS();

	//@ pure
	public NonNullResultS m1(int i) {
		return null; // OK
	} // OK

	//@ pure
	public NonNullResultS m2(int i) {
		return null; // OK
	} // OK

	//@ pure non_null
	public NonNullResultS m3(int i) {
		return null; 
	} // FAILS

	//@ pure non_null
	public NonNullResultS m4(int i) {
		return null; 
	} // FAILS

	//@ requires i >= 0;
	//@ pure non_null
	public NonNullResultS m5(int i) {
		if (i<0) return null;
		return new NonNullResultS();
	} // OK
}
