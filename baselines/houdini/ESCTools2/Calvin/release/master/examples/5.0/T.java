/* Copyright Hewlett-Packard, 2002 */

class T {

  //@ requires a != null & 0 <= i & i < a.length;
  static void storeObject(T[] a, int i, T x) {
    a[i] = x;
    }

}
