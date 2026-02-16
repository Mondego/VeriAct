/* Copyright Hewlett-Packard, 2002 */

class T4 {

  //@ requires a != null & 0 <= i & i < a.length;
  //@ requires x == null || x instanceof \elemtype(\typeof(a));
  static void storeObject(T4[] a, int i, T4 x) {
    a[i] = x;
    }

}
