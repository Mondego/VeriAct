
// This test corresponds to the bug reported in #955950
// It tests that P(o) in the code corresponds to P(o) in the annotation.
// See also M2.java - this version, which uses String, fails to establish
// an invariant that String pulls in.  I think it is a prover failure.

class M2String {
  M2String();
  boolean b;
  String obj;

  /*@ pure @*/ boolean P(String o) {
    return b;
  }

  /*@ normal_behavior
        requires P(o);
        modifies \nothing;
        ensures !\result;
      also
      normal_behavior
        requires !P(o);
        modifies o.owner;
        ensures \result;
  */
  boolean m(/*@ non_null @*/ String o) {
    boolean b;
    if (P(o)) {
      b = false;
    } else {
	//@ set o.owner = this;
	b = true;
    }
    return b;
  }
}
