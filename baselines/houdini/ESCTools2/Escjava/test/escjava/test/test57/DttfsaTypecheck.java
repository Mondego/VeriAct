class DttfsaTypecheck {
  int x;

  //@ invariant \dttfsa(int, x); // fails (expecting String literal)
  //@ invariant \dttfsa(Object[], x); // fails (expecting String literal)
  //@ invariant \dttfsa(int, "identity"); // fails (expecting 3 args)
  //@ invariant \dttfsa(int, "identity", x, x); // fails (expecting 3 args)

  //@ invariant \dttfsa(int, "f", x); // fails (expecting boolean)
  //@ invariant \dttfsa(void, "f", x); // fails (expecting boolean)
  //@ invariant \dttfsa(int, "identity", x); // fails (expecting boolean)
  //@ invariant \dttfsa(boolean, "identity", \dttfsa(int, "f", x));
  /*@ invariant \dttfsa(boolean, "EQ",
                       \dttfsa(void, "f", x),
		       \dttfsa(Object[], "g", x, x, x+x, x-x, 3, true, null));
  */

  //@ invariant \dttfsa(boolean, "f", x, y);  // fails (doesn't know y)

  void m() {
    //@ assert \dttfsa(boolean, "(EQ (fact (+ x 1)) (* (+ x 1) (fact x)))");
  }
}
