/* Copyright Hewlett-Packard, 2002 */

class WontEvenParse6 {
  void m(int x) {
    MyLabel:
    //@ assert x == x;
    x++;
  }
}
