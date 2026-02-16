class WontEvenParse4 {
  //@ ensures true;
  //@ ensures \result == (3 < 4 ? new Object() : new int[12]);
  Object m4() {
    return 3 < 4 ? new Object() : new int[12];
  }

  //@ requires (new Object()) != null;
  Object m3() {}
}
