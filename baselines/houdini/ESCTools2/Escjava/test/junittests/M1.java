
// This test is a variation of the bug reported in #955950
// It should fail because P(o) in the code may not produce the same result as P(o) 
// in the preconditions, since the state has changed.
class M1 {
  M1();
  boolean b;
  Object obj;

  /*@ pure @*/ boolean P(Object o) {
    return b;
  }

  /*@ normal_behavior
        requires P(o);
        modifies b;
        ensures !\result;
      also
      normal_behavior
        requires !P(o);
        modifies b;
        ensures \result;
  */
  boolean m(/*@ non_null @*/ Object o) {
    b = true;
    if (P(o)) {
      return false;
    }
    return true;
  }
}
