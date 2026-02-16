// This test, in combination withy AOption2, checks that options are reset
// from one testcase to the next.  See AOption1.java for a more complete
// description.

public class AOption2 {
	//@ pure
	public void m(AOption2 o) {
		o.m(this);
	}
}
