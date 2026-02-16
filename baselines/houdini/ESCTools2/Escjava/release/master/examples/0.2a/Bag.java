class Bag {
  int[] a;
  int n;

  Bag(int[] input) {
    if (input == null) {
      n = 0;
      a = new int[0];
    } else {
      n = input.length;
      a = new int[n];
      System.arraycopy(input, 0, a, 0, n);
    }
  }

  int extractMin() {
    int m = Integer.MAX_VALUE;
    int mindex = 0;
    for (int i = 1; i <= n; i++) {
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
