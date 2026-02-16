class C {
  static C x, oldx, y;
  int f;
  static int oldxf;
  static int[] a, olda, b;
  static int oldai;
  static int i;

  //@ requires x != null & y != null;
  //@ requires a != null && 0 <= i & i < a.length;
  //@ modifies oldx, oldxf, x, x.f, olda, oldai, a, a[i], i;
  //@ ensures oldx == \old(x)
  //@ ensures oldxf == \old(x.f);
  //@ ensures \old(x).f == \old(x.f) + 1;
  //@ ensures (\exists C z; z == x & \old(z.f) == \old(y.f));
  //@ ensures olda == \old(a) & oldai == \old(a[i]);
  //@ ensures \old(a)[\old(i)] == \old(a[i]) + 1;
  static void m() {
    oldx = x;
    oldxf = x.f;
    x = y;
    oldx.f++;
    olda = a;
    oldai = a[i];
    a = b;
    olda[i]++;
    i++;
  }
}
