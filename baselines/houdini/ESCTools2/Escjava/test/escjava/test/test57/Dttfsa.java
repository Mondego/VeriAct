class Dttfsa {
  void m0(int x) {
    // The following assert should fail to verify, because nothing is known
    // about function "dude".
    //@ assert (boolean)\dttfsa(boolean, "dude", x-1, x+1);
  }

  /*@ requires (\forall int a, b; a < b ==>
                                (boolean)\dttfsa(boolean, "dude", a, b)); */
  void m1(int x) {
    //@ assert (boolean)\dttfsa(boolean, "dude", x-1, x+1);
  }

  void m2(int x) {
    //@ assert \dttfsa(boolean, "is", x, \dttfsa(\TYPE, "T_int"));
  }

  void m3(int x) {
    // The following assert should fail to verify, because ESC/Java's
    // logic doesn't include anything that says that every "int" is a "double".
    //@ assert \dttfsa(boolean, "is", x, \dttfsa(\TYPE, "T_double"));
  }

  void m4(int x) {
    //@ assert \dttfsa(Object, "identity", x) == \dttfsa(Object[], "identity", x);
  }

  void m5(int x) {
    // This tests that the resulting S-expressions are predicates, not terms.
    //@ assert \dttfsa(boolean, "<", x-1, x+1);
    //@ assert !\dttfsa(boolean, "<", x-1, x+1);  // fails
  }

  void m6(int x) {
    // see m5
    //@ assert !\dttfsa(boolean, ">", x-1, x+1);
    //@ assert \dttfsa(boolean, ">", x-1, x+1);  // fails
  }

  //@ requires b;
  void m7a(boolean b) {
    // This tests that \dttfsa expressions get translated as terms, with
    // a surrounding "(EQ |@true| ...)" to make it a predicate.
    // (If this is not the case, Simplify will complain, breaking this test
    // case.)
    //@ assert (boolean)\dttfsa(boolean, "identity", b);
    //@ assert (boolean)!\dttfsa(boolean, "identity", b); // fails
  }

  //@ requires b;
  void m7b(boolean b) {
    // see m7a
    //@ assert !(boolean)\dttfsa(boolean, "identity", b); // fails
  }

  //@ requires b;
  void m8a(boolean b) {
    // see m7a
    //@ assert (boolean)\dttfsa(boolean, "identity", !b); // fails
  }

  //@ requires b;
  void m8b(boolean b) {
    // see m7a
    //@ assert !(boolean)\dttfsa(boolean, "identity", !b);
    //@ assert (boolean)!\dttfsa(boolean, "identity", !b); // fails
  }

  void m9() {
    //@ assert \dttfsa(boolean, "(EQ (+ 3 2) 5)");
  }
}
