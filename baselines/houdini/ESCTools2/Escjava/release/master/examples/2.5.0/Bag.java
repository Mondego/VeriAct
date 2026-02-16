class Bag {
  /*@ non_null */ int[] a;
  int n;
  //@ invariant 0 <= n && n <= a.length;

  //@ requires input != null;
  public Bag(int[] input) {
    n = input.length;
    a = new int[n];
    System.arraycopy(input, 0, a, 0, n);
  }

  //@ requires n >= 1; // illegal, since n may be inaccessible at call site
  public int extractMin() {
    int m = Integer.MAX_VALUE;
    int mindex = 0;
    for (int i = 0; i < n; i++) {
      if (a[i] < m) {
        mindex = i;
        m = a[i];
      }
    }
    n--;
    a[mindex] = a[n];
    return m;
  }
}
