/* Copyright Hewlett-Packard, 2002 */

class Fine {

  static Fine u;

  static Fine v;

  static int e;

  static Fine w;

  static Fine a;

  Fine f;

  static Fine b;

  static Fine c;

  static Fine d;

  static Fine x;

  static Fine y;

  Fine g;

  static Fine z;


  static void m() {
    if (u != null | v != null) e = 10;
    if (v == null) {
      w = a.f;
      //@ assert b != null;
      //@ assume a != null & b != null & c != null & d != null & e > 0;
      x = c.f;
      //@ assert d != null;
    } else {
      c = new Fine();
      d = v;
    }
    y = a.g;
    //@ assert b != null;
    z = c.g;
    //@ assert d != null;
    //@ assert e > 0;
  }


}

