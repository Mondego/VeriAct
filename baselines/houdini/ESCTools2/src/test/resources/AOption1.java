// This test tests whether flags get reset for each testcase when running
// JUnit tests.  JUnit tests are run sequentially within one execution of
// the escjava program.  If the options are static and not reset (as was
// originally the case in escjava) then options set for one testcase will
// persist for the remainder.  That behavior has been changed so that the
// options are reset for each test case.  This test (AOption1) uses some
// command-line options; it is followed by AOption2 which does not use those
// options; if the options persist, AOption2 will have an incorrect result.
// There still may be options that need individual resetting that are not
// tested here.

//#FLAGS: -quiet -nowarn Null

public class AOption1 {
	//@ pure
	public void m(AOption1 o) {
		o.m(this);
	}
}
