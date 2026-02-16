
// This test corresponds to the bug reported in #955950
// It tests that P(o) in the code corresponds to P(o) in the annotation.
class M2 {
  M2();
  boolean b;
  SS obj;

  /*@ pure @*/ boolean P(SS o) {
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
  boolean m(/*@ non_null @*/ SS o) {
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

class SS {
  SS();
}
