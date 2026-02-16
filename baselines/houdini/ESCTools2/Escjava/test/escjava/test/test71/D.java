class D {
  boolean b, c, d;

  // The following are not allowed
  //@ invariant b ==> c <== d;
  //@ invariant b <== c ==> d;
  //@ invariant b ==> b <== b <== b ==> b ==> b <== b <== b ==> b ==> b <== b;
}
