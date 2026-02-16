
// This test corresponds to the bug reported in #955950
// It tests that P(o) in the code corresponds to P(o) in the annotation.
class M {
  M();
  boolean b;
  Object obj;

  /*@ pure @*/ boolean P(Object o) {
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
  boolean m(/*@ non_null @*/ Object o) {
    if (P(o)) {
      return false;
    }
    //@ set o.owner = this;
    return true;
  }
}
