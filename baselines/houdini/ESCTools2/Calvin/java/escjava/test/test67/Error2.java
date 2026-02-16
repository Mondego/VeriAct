/* Copyright Hewlett-Packard, 2002 */

class Error2 {
  // The following is not allowed, because trailing @-signs
  // are ignored only in "/*"-comments.

  //@ requires x instanceof Error2    @@@@@
  void r(/*@@@@@@@@ non_null @@@@@*/ Object x) {
  }
}
