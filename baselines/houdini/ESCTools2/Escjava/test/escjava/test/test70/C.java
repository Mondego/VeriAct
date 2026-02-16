class C {
  void a() {
    m();
    p();
    //@ assert ! (* this should generate a warning *);
  }

  //@ requires (* this is an informal predicate *);
  //@ requires true &&(*soIsThis*)
  //@ ensures (((((* Here's one inside some parens *)))));
  void m() {
  }

  //@ requires (* this has what looks like a // comment inside, but it isn't *)
  /*@ requires (* this one takes up more
                         than one line *); */
  //@ requires (* one can even /* do this!! */;  cool, huh? *);
  //@ requires (* and /* this!! *);
  //@ ensures (**);
  /** This is a Javadoc comment <esc>ensures !!(*nothing fancy, really*)</esc>
   **/
  void p() {
  }

  /*@ requires (\forall int j; (* here's the range of j *); (* and
         here is the term *) && b[j] && (* and some more *)); */
  void q(int x, /*@ non_null */ boolean[] b) {
  }
}
