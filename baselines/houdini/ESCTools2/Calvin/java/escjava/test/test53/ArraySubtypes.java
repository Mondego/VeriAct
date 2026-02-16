/* Copyright Hewlett-Packard, 2002 */

class ArraySubtypes {
  //@ requires a != null && a.length != 0;
  //@ requires a[0] != null && a[0].length != 0;
  //@ requires \typeof(b) <: \elemtype(\typeof(a[0]));
  void m0(Object[][] a, Object b) {
    a[0][0] = b;
    // The following assert should fail because "a" may have
    // been equal to "a[0]" initially.
    /*@ assert a[0][0] == b; */  // this should fail
  }

  //@ requires a != null && a.length != 0;
  //@ requires a[0] == a;
  //@ requires \typeof(b) <: \elemtype(\typeof(a));
  void m1(Object[][] a, Object b) {
    //@ assert \typeof(a) <: \elemtype(\typeof(a));
    a[0] = a;
    a[0][0] = b;
    /*@ assert a[0][0] == b; */  // this should fail
  }

  //@ requires a != null && a.length != 0;
  //@ requires \typeof(a) <: \elemtype(\typeof(a));
  //@ requires \typeof(b) <: \elemtype(\typeof(a));
  void m2(Object[][] a, Object b) {
    a[0] = a;
    a[0][0] = b;
    /*@ assert a[0][0] == b; */  // this should fail
  }

  //@ requires a != null && a.length != 0;
  //@ requires \typeof(a) == \type(Object[][]);
  //@ requires \typeof(b) <: \elemtype(\typeof(a));
  void m3(Object[][] a, Object b) {
    //@ assert \typeof(a) <: \elemtype(\typeof(a));
    a[0] = a;
    a[0][0] = b;
    /*@ assert a[0][0] == b; */  // this should fail
  }

  //@ requires a != null && a.length != 0;
  //@ requires \typeof(a) == \type(Object[][]);
  //@ requires \typeof(b) <: \type(Object[]);
  void m4(Object[][] a, Object b) {
    /* This example shows exactly how "m0()" may fail */
    a[0] = a;
    a[0][0] = b;
    //@ assert a[0] == b;
    /*@ assert a[0][0] == b; */  // this should fail
  }

  //@ requires a != null && a.length != 0;
  //@ requires a[0] != null && a[0].length != 0;
  //@ requires \typeof(b) <: \elemtype(\typeof(a[0]));
  void m5(Object[][] a, Object b) {
    if (a != a[0] || a[0] == b) {
      a[0][0] = b;
      /*@ assert a[0][0] == b; */  // this should succeed
    }
  }

  void m6(Object b) {
    Object[][] a = new Object[12][40];
    a[0][0] = b;
    /*@ assert a[0][0] == b; */  // this should succeed
  }
}
