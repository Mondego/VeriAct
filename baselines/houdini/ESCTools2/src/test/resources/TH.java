// This tests that the use of 'this' in constructor preconditions 
// is translated ok

public class TH {
	static TH a,b;

        int i = 0;

	//@ modifies \nothing;
	public TH() {}

	//@ requires x != null;
	//@ requires x != this;
	//@ modifies \nothing;
        //@ ensures x != this;
	public TH(TH x) {
	    //@ assert a != this && x != this;
	}


	public static void m() {
		TH z = new TH();
		TH y = new TH(z);
		//@ assert y != z;
	}

        public void mm() {
		TH y = new TH(new TH());
		//@ assert y != this;
	}


        //@ requires i == 0;
	//@ ensures i == k;
        public TH(int k) {
		i = k;
	}
}
