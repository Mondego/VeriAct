class verbose {
  int x;
  //@ invariant x == 0;

  void m(int y) {
    x = y;
    n(y);
  }

  /*@ helper */ private void n(int y) {
  }

  void p(int y) {
    m(y);
    x = y+1;
  }
}
