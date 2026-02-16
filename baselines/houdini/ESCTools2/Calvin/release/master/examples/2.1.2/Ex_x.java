/* Copyright Hewlett-Packard, 2002 */

class Ex_x {

  static int g, h, i, j;

  static boolean b;

  static void m(int a) {














    // start complicated computation guaranteed to leave i != 0

    i = a*a + 1;  // 

























































































































    // end of complicated computataion
    // If we don't assume i != 0 then ESC/Java can't figure it out.
    if (b) {
      g = h/i;
    } else {
      h = g/i + g/j;
    }
  }
}

