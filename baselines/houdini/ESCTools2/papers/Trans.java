public class Trans {
  public static class C {
    public int z;
  }
  boolean b;

  //@ diverges false;
  //@ ensures b || \result == (c.z == 1);
  //@ signals (Exception) false;
  //@ pure
  public boolean p(C c, String s);

  //@ requires p(c,"A");               // A
  //@ modifies b;
  //@ ensures p(c,"Z");                // Z
  public void m(C c) {
    //@ assert p(c,"B");               // B
    b = p(c,"Q");                      // Q
    //@ assert p(c,"C") && p(c,"D");   // C
    c = new C();
    //@ assert p(c,"E");               // E
  }
}
