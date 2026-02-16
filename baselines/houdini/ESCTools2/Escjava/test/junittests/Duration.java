// Tests what happens when space and duration functions are used

public class Duration {

	public Object o;

	//@ ensures \space(o) > 0;
	void m() {}

	//@ ensures \duration(m()) > 0;
	void mm() {}

	//@ ensures \working_space(new Duration()) > 0;
	void mmm() {}
}
