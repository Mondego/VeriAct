/* Copyright Hewlett-Packard, 2002 */

class T2 {

  //@ requires a != null & 0 <= i & i < a.length;
  //@ requires \typeof(a) == \type(T2[]);
  static void storeObject(T2[] a, int i, T2 x) {
    a[i] = x;
    }

}
