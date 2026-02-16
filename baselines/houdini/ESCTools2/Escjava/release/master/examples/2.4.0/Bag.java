class Bag { 
  int[] a; 
  //@ invariant a != null; 
  int n;
  //@ invariant 0 <= n && n <= a.length;

  Bag(/*@ non_null */ int[] input) {
    n = input.length;
    a = new int[n];
    System.arraycopy(input, 0, a, 0, n);
  }

  //@ requires n >= 1;
  int extractMin() {
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
