// Tests defaults in behavior specs
//#FLAGS: -sourcepath .

public class Behavior {

	/*@ public exceptional_behavior
		signals (java.lang.Exception e) true;
	*/
	public void m() throws Exception {
		throw new Exception();
	}

	/*@ public exceptional_behavior
		signals (java.lang.Exception e) true;
	*/
	public void m2() throws Exception { // WARNING HERE
		return;
	}

	/*@ public normal_behavior
		ensures true;
	*/
	public void m3() {
		return;
	}

	/*@ public normal_behavior
		ensures true;
	*/
	public void m4() throws Exception {
		throw new Exception();
	}

	/*@ public normal_behavior
		requires i>0;
		ensures true;
	 also
	    public exceptional_behavior
		requires i<0;
		signals (java.lang.Exception e) true;
	*/
	public void mm(int i) throws Exception {
		if (i>0) return;
		if (i<0) throw new Exception();
		return;
	}

	/*@ public normal_behavior
		requires i<0;
		ensures true;
	 also
	    public exceptional_behavior
		requires i>0;
		signals (java.lang.Exception e) true;
	*/
	// This one produces warnings
	public void mme(int i) throws Exception {
		if (i>0) return;
		if (i<0) throw new Exception();
		return;
	}

	public int j;
	//@ modifies j;
	//@ ensures j < \old(j); // Warning here
	public void n() { ++j; }
}
