// Tests the combining of super class and subclass specs


public class InheritedSpecs extends ISHelper {

	//@ also
	//@ requires i >= 0;
	//@ ensures \result >= 0;
	/*@ pure */ public int m(int i, int j) {
		return i;
	} // FAILS

	public void mm1() { m(1,1); } // OK
	public void mm2() { m(1,-1); } // OK
	public void mm3() { m(-1,1); } // OK
	public void mm4() { m(-1,-1); } // FAILS
	public void mm5() { (new ISHelper()).m(1,1); } // OK
	public void mm6() { (new ISHelper()).m(1,-1); } // FAILS
	public void mm7() { (new ISHelper()).m(-1,1); } // OK
	public void mm8() { (new ISHelper()).m(-1,-1); } // FAILS

	//@ also
	//@ requires i > 0;
	/*@ pure */ public void n(int i);

	public void nn() { 
		n(-1);  // OK
	}

	/*@ pure */ public InheritedSpecs();


	public void n1(int i);
	public void n2(int i);
	public void n3(int i);

	//@ also
	//@ requires i%2 == 0;
	public void n4(int i);
	//@ also
	//@ requires i%2 == 0;
	public void n5(int i);
	//@ also
	//@ requires i%2 == 0;
	public void n6(int i);
	//@ also
	//@ requires i%2 == 0;
	public void n7(int i);

	public void nn1() {
		n1(0); // OK
		n1(2); // OK
	}
	public void nn1e() {
		n1(-1); // FAILS
	}

	public void nn2() {
		n2(2); // OK
		n2(0); // OK
		n2(-1); // OK
		n2(-20); // OK
	}

	public void nn3() {
		n2(2); // OK
		n2(0); // OK
		n3(-20); // OK
	}
	public void nn3e() {
		n3(-1); // FAILS
	}

	public void nn4() {
		n4(0); // OK
		n4(1); // OK
	}

	public void nn5() {
		n5(0); // OK
		n5(1); // OK
		n5(-2); // OK
	}
	public void nn5e() {
		n5(-1); // FAILS
	}

	public void nn6() {
		n6(3); // OK
		n6(2); // OK
		n6(-1); // OK
		n6(-2); // OK
	}

	public void nn7() {
		n7(0); // OK
		n7(3); // OK
		n7(2); // OK
		n7(1); // OK
		n7(-2); // OK
		n7(-20); // OK
		n7(-21); // OK
	}
	public void nn7e() {
		n7(-1); // FAILS
	}
}

class ISHelper extends ISSHelper {

	//@ requires j >= 0;
	//@ ensures \result >= 0;
	/*@ pure */ public int m(int i, int j) { return 1; }

	// Test that no specs are inherited as 'requires true'
	/*@ pure */ public void n(int i);

	/*@ pure */ public ISHelper();

	public void n1(int i);
	//@ also
	//@ requires i>1 || i< -10;
	public void n2(int i);
	//@ also
	//@ requires i>1 || i< -10;
	public void n3(int i);
	public void n4(int i);
	public void n5(int i);
	//@ also
	//@ requires i>1 || i< -10;
	public void n6(int i);
	//@ also
	//@ requires i>1 || i< -10;
	public void n7(int i);
}

class ISSHelper {
	/*@ pure */ public ISSHelper();

	//@ requires i>=0;
	/*@ pure */ public void n1(int i);

	/*@ pure */ public void n2(int i);
	//@ requires i>=0;
	/*@ pure */ public void n3(int i);
	/*@ pure */ public void n4(int i);
	//@ requires i>=0;
	/*@ pure */ public void n5(int i);
	/*@ pure */ public void n6(int i);
	//@ requires i>=0;
	/*@ pure */ public void n7(int i);
}

