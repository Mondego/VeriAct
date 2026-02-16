class C {

  int i;
  int[] x;
  /*@ invariant i > 0
      ==> x != null
  */

  void m (int[] p, int[] q) {
    i = 10;
    int[] t;
    if (p != null) {
      t = p;
    } else {
      t = q;
    }
    x = t;
  }
}
