class Error1 {
  // The following is not allowed, because additional @-signs
  // are ignored only in "/*"-comments.

  //@@@@@@@@ requires x instanceof Error1
  void r(/*@@@@@@@@ non_null @@@@@*/ Object x) {
  }
}
