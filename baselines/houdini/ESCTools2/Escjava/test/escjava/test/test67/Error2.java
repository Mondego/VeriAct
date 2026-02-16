class Error2 {
  // The following is not allowed, because trailing @-signs
  // are ignored only in "/*"-comments.
  // Actually - per the JML design doc - it is allowed, and has been fixed.
  //@ requires x instanceof Error2    @@@@@
  void r(/*@@@@@@@@ non_null @@@@@*/ Object x) {
  }
}
