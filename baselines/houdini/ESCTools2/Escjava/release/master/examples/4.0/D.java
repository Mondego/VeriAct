class D {

  /*@ non_null */ static int[] x = new int[10];

  static void m(int[] a, int[] b) {
    int[] c;
    if (a != null) {
      c = a;
    } else {
      c = b;
    }
  x = c;
  }
}
