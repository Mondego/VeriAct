class Ex {

  static int g, h, i, j;

  static boolean b;

  static void m(int a) {














    // start complicated computation guaranteed to leave i != 0

    i = a*a + 1;  // 

























































































































    // end of complicated computataion
    //@ assume i != 0l;
    if (b) {
      g = h/i;
    } else {
      h = g/i + g/j;
    }
  }
}

