/* Copyright Hewlett-Packard, 2002 */

class D {

  int a, b;
  //@ invariant (\lblpos foo a >= 0 & b >= 0) || (\lblpos bar a < 0 & b <0);

  //@ requires x != null && y != null;
  D(D x, D y) {
    a = x.a;
    b = y.b;
  }

}
