package eu.etaps.tutorial.bags;

class Bag1 {
  /*@ non_null */ int[] a;
  int n;
  //@ invariant 0 <= n && n <= a.length;
  //@ ghost public boolean empty;
  //@ invariant empty == (n == 0);

  //@ requires input != null;
  //@ ensures this.empty == (input.length == 0);
  public Bag1(int[] input) {
    n = input.length;
    a = new int[n];
    System.arraycopy(input, 0, a, 0, n);
    //@ set empty = n == 0;
  }

  //@ ensures \result == empty;
  public boolean isEmpty() {
    return n == 0;
  }

  //@ requires !empty;
  //@ modifies empty;
  //@ modifies n, a[*];
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
    //@ set empty = n == 0;
    //@ assert empty == (n == 0);
    a[mindex] = a[n];
    return m;
  }
}
