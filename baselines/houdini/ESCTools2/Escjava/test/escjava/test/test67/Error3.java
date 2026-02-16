class Error3 {
  // The following is not allowed, because additional @-signs
  // on the first line of a "/*"-comment cannot have any intervening
  // white space.

  /*@ @@@@@@  requires x instanceof Error3 */
  // ^ This is the problem.
  void r(Object x) {
  }
}
