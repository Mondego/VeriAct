/* Copyright Hewlett-Packard, 2002 */

class TryFinally {
  void m() {
    int x,y;
    try {
      x=1;
    } finally {
      x=2;
    }
    //@ assert x==2;
  }
}
