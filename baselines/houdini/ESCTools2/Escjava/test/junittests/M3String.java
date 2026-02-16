
// This test corresponds to the bug reported in #955950
// It tests that P(o) in the code corresponds to P(o) in the annotation.
// See also M2.java - this version, which uses String, fails to establish
// an invariant that String pulls in.  I think it is a prover failure.

public class M3String {
  M3String();
  String obj;

  //@ modifies o.owner;
  void m(/*@ non_null @*/ String o) {
	//@ set o.owner = this;
        //@  assume o.charArray.length == \old(o.charArray.length);
  }
}
