
public class THERR {
	static TH a,b;

        int i = 0;

        //@ requires i == 0;  // OK
        public THERR(int k) {
	}

        //@ requires i == 1; 
	public THERR(int k, int kk) {}

	int ii = 1;

	//@ requires ii == 1;
	public THERR() {}

	public void p() {
		i = 0;
		THERR z = new THERR(0);
	}
	public void p1() {
		i = 1;
		THERR z = new THERR(0); // SHOULD FAIL
	}
	public void p2() {
		THERR z = new THERR(0);
	}

	public void q() {
		ii = 0;
		THERR z = new THERR(); // SHOULD FAIL
	}
	public void q1() {
		ii = 1;
		THERR z = new THERR();
	}
	public void q2() {
		THERR z = new THERR();
	}

	public void m() {
		i = 0;
		THERR z = new THERR(0,0); // SHOULD FAIL
	}
	public void m1() {
		i = 1;
		THERR z = new THERR(0,0);
	}
	public void m2() {
		THERR z = new THERR(0,0); // SHOULD FAIL
	}
}
