// Tests that precondition failures are located on the correct declartion

public class PreconditionWarnings {

	//@ requires i < 0;
	//@ requires j < 0;
	//@ also
	//@ requires i > 0;
	//@ requires j > 0;
	/*@ pure */ public void m(int i, int j) {}

	public void mm() {
		m(1,0);
	}
	public void mm1() {
		m(0,1);
	}
	public void mm2() {
		m(-1,0);
	}
	public void mm3() {
		m(0,-1);
	}
	public void mm4() {
		m(1,1);
	}
	public void mm5() {
		m(-1,-1);
	}
	public void mm6() {
		m(1,-1);
	}
}

