/* Copyright Hewlett-Packard, 2002 */

class T1 {

  //@ requires a != null & 0 <= i & i < a.length;
  //@ requires \elemtype(\typeof(a)) == \type(T1);
  static void storeObject(T1[] a, int i, T1 x) {
    a[i] = x;
    }

}
