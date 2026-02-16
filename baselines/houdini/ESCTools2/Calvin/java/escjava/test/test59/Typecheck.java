/* Copyright Hewlett-Packard, 2002 */

class Typecheck {
  static void p() { }
  void m() { }
  static int x;
  int y;

  //@ non_null
    { p(); }

  //@ requires x < y;
  //@ ensures x + y + z;  // this doesn't even type check, but that's not checked
    { m(); }

  //@ modifies x < y;
  //@ ensures x + y + z;  // this doesn't even type check, but that's not checked
  static {
    p();
    m();  // can't do this in static initializer block
    y++;  // can't do this in static initializer block
    x++;
  }

  //@ also_ensures true;
  //@ nowarn Invariant;
  //@ monitored
    { y++; x++; m(); p(); }

  //@ readable_if false;
  //@ monitored
  //@ monitored_by x << true;
  static { int z; z++; }
}
