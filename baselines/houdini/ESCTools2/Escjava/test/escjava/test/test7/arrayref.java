
class arrayref {
  public static int m1(int[] a) {
    //@ assume a != null;
    //@ assume a.length == 10;
    return a[1];
  }

  public static int m2(int[] a) {
    //@ assume a != null;
    //@ assume a.length == 10;
    int x = 1;
    return a[x] + a[x += 1];
  }
}
