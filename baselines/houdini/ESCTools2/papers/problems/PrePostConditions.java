package problems;

class PrePostConditions
{
  byte b;
  int i;
  Object o;
  String s;
  static Object so;
  
  void m() {
    o = s.toString();
  }
  
  void n() {
    b = (byte)i;
    if (b == i)
      s = new String();
    //@ assert s != null;
  }
  
  void o(int j) {
    if (j == b)
      o = null;
    //@ assert o != null;
  }
  
  void p(String t) {
    PrePostConditions.so = t.substring(3,6);
  }
  
  void q() {
    if (StaticPreconditions.o.hashCode() == 0) {
      StaticPreconditions.i = 1;
    }
    assert i == 1;
  }
  
  void r(Object o, byte b) {
    if (o == null)
      throw new IllegalArgumentException();
    if (b < 0)
      throw new IllegalArgumentException("bogus byte");
    i = 2;
  }
  
  public static void main(String[] args) {
    PrePostConditions ppc = new PrePostConditions();
    ppc.m();
    ppc.n();
    ppc.o(127);
    ppc.p("foobar");
    ppc.q();
  }
}

class StaticPreconditions {
  static int i;
  static Object o;
}
