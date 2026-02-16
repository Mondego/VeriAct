class C {
  boolean[] b;
  int[] x;

  //@ requires (\forall int j; 0 <= j && j < b.length ==> b[j]);
  void m0() {
    m1();
  }

  //@ requires (\forall int j;; 0 <= j && j < b.length ==> b[j]);
  void m1() {
    m2();
  }

  //@ requires (\forall int j; 0 <= j && j < b.length; b[j]);
  void m2() {
    m0();
  }

  //@ requires (\exists int j; 0 <= j && j < b.length && b[j]);
  void p0() {
    p1();
  }

  //@ requires (\exists int j;; 0 <= j && j < b.length && b[j]);
  void p1() {
    p2();
  }

  //@ requires (\exists int j; 0 <= j && j < b.length; b[j]);
  void p2() {
    p0();
  }

  //@ requires (\forall int j; b[j] && x[j] > 0);
  void w0() {
    w1();
  }

  //@ requires (\forall int j; b[j]; x[j] > 0);
  void w1() {
    w0();  // error
  }

  //@ requires (\exists int j; b[j] ==> x[j] > 0);
  void w2() {
    w3();  // error
  }

  //@ requires (\exists int j; b[j]; x[j] > 0);
  void w3() {
    w2();
  }
}
