/* Copyright Hewlett-Packard, 2002 */

class Simple0 {
  Simple0() {
  }

  int x;
  
  void m() {
    x = 5;
    int y = n();
    //@ assert y == 5;
  }

  private int n() {
    //@ assert 0 <= x;
    return x;
  }
}
