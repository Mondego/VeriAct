class WontTypecheck {
  //@ requires x < (* this should be an integer but isn't *);
  //@ requires (* this is fine *) ? (* but this is a boolean *) : x;
  //@ requires (* similarly *) ? x : (* now the boolean is here *);
  void m(int x) {
  }
}
