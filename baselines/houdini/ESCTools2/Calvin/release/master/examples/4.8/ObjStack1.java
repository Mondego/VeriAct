/* Copyright Hewlett-Packard, 2002 */

class ObjStack {
  /*@ non_null */ Object [] a;
  //@ invariant a.length == 10;
  //@ invariant \elemtype(\typeof(a)) == \type(Object);
  int n;  //@ invariant 0 <= n & n <= 10;
  //@ invariant (\forall int i; n <= i & i < 10 ==> a[i] == null);
  //@ invariant a.owner == this;

  ObjStack() {
    n = 0;
    a = new Object[10];
  }

  //@ requires n < 10;
  void Push(Object o) {
    a[n++] = o;
  }

  //@ requires n > 0;
  Object Pop() {
    Object o = a[--n];
    a[n] = null;
    return o;
  }

}
