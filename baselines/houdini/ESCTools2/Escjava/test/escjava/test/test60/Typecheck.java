// This test concerns the use of preconditions and non_null parameter
// pragmas with instance methods.
//
// A method is "class-new" if it is declared in a class and does not
// override any method of any superclass.  A method is "new" if it doesn't
// override any method anywhere.
//
// Rules about non_null:
//   A non_null pragma may be place on an appropriately typed
//   parameters in "new" methods.
//
// Rules about requires:
//   A method can have a "requires" clause only if it is a new method.
//
// Rules about also_requires:
//   A method can have an "also_requires" clause only if it is a class-new
//   method that is not also a new method.

class Super {
  public void m0(/*@ non_null */ int[] x) {
  }
  public void m1(int[] x) {
  }
  public void m2(int[] x) {
  }

  //@ requires 0 <= k;
  public void p0(int k) {
  }
  //@ requires 0 <= k;
  public void p1(int k) {
  }
  /*@ also requires 0 <= k; */  // error: cannot put "also requires" here
  public void p2(int k) {
  }
}

interface J {
  public void m0(int[] x);
  public void m1(int[] x);
  public void m2(/*@ non_null */ int[] x);
  public void m4(int[] x);
  public void m5(/*@ non_null */ int[] x);

  //@ requires 0 <= k;
  public void p0(int k);
  //@ requires 0 <= k;
  public void p1(int k);
  /*@ also requires 0 <= k; */  // error: cannot put "also requires" here
  public void p2(int k);
  /*@ also requires 0 <= k; */  // error: cannot put "alsorequires" here
  public void p3(int k);
  public void p4(int k);
}

class Typecheck extends Super implements J {
  public void m0(/*@ non_null */ int[] x) { // error: method is not new
  }
  public void m1(/*@ non_null */ int[] x) { // error: method is not new
  }
  public void m2(/*@ non_null */ int[] x) { // error: method is not new
  }
  public void m3(/*@ non_null */ int[] x) { // fine:  method is new
  }
  public void m4(/*@ non_null */ int[] x) { // error:  method is class-new, not new
  }
  public void m5(/*@ non_null */ int[] x) { // error:  method is class-new, not new
  }

  /*@ requires 0 <= k; */   // error: cannot put "requires" here
  public void p0(int k) {
  }
  public void p1(int k) {  // fine
  }
  /*@ also requires 0 <= k; */  // error: cannot put "also requires" here
  public void p2(int k) {
  }
  /*@ also requires 0 <= k; */  // fine:  class-new methods are where
                                // "also requires" belong
  public void p3(int k) {
  }
  /*@ requires 0 <= k; */   // error: cannot put "requires" here (but
                            // "also requires" would work)
  public void p4(int k) {
  }
}

interface K extends J {
  public void p(/*@ non_null */ int[] x);  // fine: method is new
  public void m0(/*@ non_null */ int[] x);  // error:  method is not new (or class-new)
  public void m2(/*@ non_null */ int[] x);  // error:  method is not new (or class-new)

  /*@ requires 0 <= k; */   // error: cannot put "requires" here
  public void p0(int k);
  public void p1(int k);  // fine
  /*@ also requires 0 <= k; */  // error: cannot put "also requires" here
  public void p2(int k);
  /*@ also requires 0 <= k; */  // error: cannot put "also requires" here
  public void p3(int k);
}
