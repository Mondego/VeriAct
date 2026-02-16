class C {

  //@ requires (\lblpos fee i < 5) || (\lblpos fie i > 10);
  //@ ensures (\lblneg foe \result != 5) && (\lblneg fum \result > 0);
  int m(int i) {
      return i;
  }

}
