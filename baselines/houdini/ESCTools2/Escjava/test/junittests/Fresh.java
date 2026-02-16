// Tests the \fresh function

public class Fresh {

	public Object o = new Object();

	//@ ensures \fresh(\result);
	public Object m() {
		return new Object();
	}

	//@ ensures \fresh(o);
	public void mm() {} // FAILS

	//@ ensures \fresh(new Object());
	public void mmm() {}
}

