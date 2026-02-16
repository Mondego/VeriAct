class T3 {

  //@ requires a != null & 0 <= i & i < a.length;
  //@ requires \typeof(a) == \type(T3)[];
  static void storeObject(T3[] a, int i, T3 x) {
    a[i] = x;
    }

}
