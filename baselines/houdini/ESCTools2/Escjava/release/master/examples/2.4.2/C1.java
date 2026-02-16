class C1 {

  //@ axiom (\forall int x, y; x >= 0 & y >= 0 ==> x*y >= 0);

  //@ requires i >= 0;
  //@ ensures \result >= 0;
  static int s(int i) {
    return i * i;
  }

}
