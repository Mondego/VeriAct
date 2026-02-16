public class ArrayOps {

  private /*@ spec_public @*/ Object[] a;

  //@ public invariant 0 < a.length;

  /*@ requires 0 < arr.length;
    @ ensures this.a == arr;
    @*/
  public void init(Object[] arr) {
    this.a = arr;
  }

  /*@ requires 0 <= i && i < a.length;
    @ ensures a[i] == nu
    @     && \result == \old(a[i]);
    @*/
  public Object replace(int i, Object nu) {
    Object old = a[i];
    a[i] = nu;
    return old;
  }
}
