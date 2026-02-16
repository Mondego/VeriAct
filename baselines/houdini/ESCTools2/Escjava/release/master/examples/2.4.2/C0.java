class C0 {

  //@ requires i >= 0;
  //@ ensures \result >= 0;
  static int sq(int i) {
    return i * i;
  }

}
