// Tests that \duration and \working_space functions are parsed

public class ParseDuration {

	//@ working_space \space(this);
	//@ working_space \space(this) if true;
	public void m();

	//@ duration \duration(m());
	//@ duration \duration(m()) if false;
	public void n(int i);

	//@ invariant \duration(m()) < 10;
	//@ invariant \working_space(n(0)) < 10;
	//@ invariant \space(this) < 10;

	// Errors:
	//@ invariant \duration() < 10;
	//@ invariant \duration(m,n) < 10;
	//@ invariant \working_space() < 10;
	//@ invariant \working_space(m,n) < 10;
}
