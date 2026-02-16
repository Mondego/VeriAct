/* Copyright Hewlett-Packard, 2002 */

class BogusSpec {
  //@ ensures \result < 6;
  //@ ensures \result == \result;
  //@ ensures \result != \result;
  //@ ensures \result <= \result;
  void m0() {
  }

  Object x;
  //@ requires x == 6;
  //@ modifies this != null;
  //@ modifies this.x, x
  //@ ensures this.x == null || \fresh(this.x);
  //@ ensures \result < x;
  int m1(int x) {
  }

  //@ requires \result;
  //@ modifies \result
  //@ ensures \result;
  boolean m2() {
  }

  //@ requires true;
  //@ modifies x, x
  //@ also_modifies x, x
  //@ ensures x == null;
  //@ also_ensures x == null;
  void m3() {
  }

  //@ modifies a
  //@ requires 0 <= p && a != null;
  //@ modifies a[*], a, a[p]
  //@ ensures \old(a[p]) == p ==> a[p+1] == p+1;
  void m4(int[] a, int p) {
  }

  //@ requires x != null;
  //@ requires this != null;
  //@ modifies \result.x;
  //@ modifies \result, x;
  //@ ensures \result == null;
  static Object m5() {
    return null;
  }

  //@ requires \lockset == \lockset;
  //@ requires \lockset;
  //@ modifies \lockset
  //@ ensures \lockset == \lockset;
  //@ ensures \lockset;
  void m6() {
  }

  //@ requires \old(x) == x;
  //@ requires \old(\old(x)) == x;
  //@ modifies x;
  //@ modifies \old(this).x;
  //@ ensures \old(x) == x;
  //@ ensures \old(this) == null;
  //@ ensures \old(x==x && \old(this)==this);
  void m7() {
  }

  //@ requires \fresh(x);
  //@ ensures \fresh(x) == \fresh(this);
  //@ ensures \fresh(\fresh(this) ? x : x);
  //@ ensures \fresh(\old(x));
  //@ ensures \old(\fresh(x));
  void m8() {
  }
  
  //@ requires x != null;
  //@ requires this != null;
  //@ requires \result != null;
  //@ ensures \result != null;
  //@ ensures this != null;
  //@ modifies \result.x
  //@ modifies x, this.x;
  BogusSpec(int a) {
  }

  //@ ensures \result == null;
  //@ exsures (Throwable t) \result == null
  //@ exsures (Throwable t) t == null
  //@ exsures (SomeException se) se == \result
  //@ exsures (SomeException se) se == x && se == this;
  //@ also_exsures (Throwable t) t == null
  //@ exsures (Object o) true; exsures (int t) false;
  //@ exsures (Throwable x) true;
  //@ exsures (Throwable) true; 
  //@ exsures (java.lang.Throwable th) true;
  //@ exsures (java.lang.Throwable) true;
  //@ exsures (int) true;
  //@ exsures (Throwable j) true;
  //@ exsures (Throwable t) (\forall Throwable t; t == t);
  //@ exsures (\TYPE tt) tt == \type(SomeException);
  //@ exsures (int[] a) a == a;
  //@ exsures (Cloneable c) true;
  Object m9(int j) throws Throwable {
    return null;
  }

  //@ exsures (SomeException se) se != null;
  //@ exsures (Throwable t) t == 5;
  //@ exsures (SomeSubException sse) false;
  //@ exsures (SomeOtherException soe) soe != null;
  //@ exsures (RuntimeException rte) true;
  //@ exsures (ArrayStoreException ase) ase == ase;
  void m10() throws SomeException, NullPointerException, AnotherException {
  }

  //@ exsures (SomeException se) \result == 2;
  //@ exsures (SomeException se) this != x;
  //@ also_exsures (SomeException se) true
  static int m11() throws SomeException {
    return 4;
  }

  //@ exsures (SomeException se) se == \result && se == this && se == x;
  //@ exsures (Throwable tt) tt == se;
  //@ exsures (Throwable tt) d < 0.0;
  //@ exsures (SomeOtherException soe) true;
  //@ also_exsures (Throwable tt) true
  BogusSpec(double d) throws SomeException {
  }
}

class BogusSpecSub extends BogusSpec {
  //@ requires true;
  //@ modifies x, x
  //@ also_modifies x, x
  //@ ensures x == null;
  //@ also_ensures x == null;
  void m3() {
  }

  //@ also_modifies \lockset
  //@ also_ensures \lockset == \lockset;
  //@ also_ensures \lockset;
  void m6() {
  }

  //@ exsures (Throwable t) true;
  //@ also_exsures (SomeException se) se != null;
  //@ also_exsures (Throwable t) t == 5;
  //@ also_exsures (SomeSubException sse) false;
  //@ also_exsures (SomeOtherException soe) soe != null;
  //@ also_exsures (NullPointerException npe) true;
  //@ also_exsures (\TYPE tt) tt == \type(SomeException);
  //@ also_exsures (AnotherException ae) ae == null;
  void m10() throws SomeSubException, AnotherException {
  }
}

class SomeException extends Throwable {
}

class SomeOtherException extends Throwable {
}

class SomeSubException extends SomeException {
}

class AnotherException extends RuntimeException {
}

class FClass {
  int y = 4;
  final int x = 5;
  static final boolean b = false;
  final int[] a;

  //@ modifies x, this.x, this.b, FClass.b;
  //@ modifies b, y;
  //@ modifies a, a.length, a[4], a[*]
  void m() {
  }

  //@ modifies b, f.x;
  static void p(FClass f) {
  }
}

class FFClass extends FClass {
  final int xx = 5;
  static final boolean bb = false;
  
  //@ also_modifies this.xx;
  //@ also_modifies x, y, b;
  //@ also_modifies bb;
  void m() {
  }
}
