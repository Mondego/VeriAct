class Methods {
  /*@ non_null */ /*@ non_null */ /*@ non_null */ String m(String s) {
    return s;
  }  // warning: possibly null

  //@ requires x == 0 ==> r != null;
  /*@ non_null */ Object p(int x, /*@ non_null */ String t, Object r) {
    switch (x) {
      case 0:  return r;
      default:  return t;
    }
  }
}

interface J {
  public /*@ non_null */ Object h();
}

interface K0 extends J {
}

interface K1 extends J {
}

interface L extends K0, K1 {
}

class M implements L {
  public Object h() {
    return null;
  }  // error:  null return
}

class N extends Methods {
  Object p(int x, String t, Object r) {
    return r;
  }  // possible error:  null return
}

class O extends N implements K1 {
  Object p(int x, String t, Object r) {
    return new Object();
  }
  public Object h() {
    return p(5, "hello", null);
  }
}

class NowarnMe {
  /*@ non_null */ Object y0() {
    return null;
  }  //@ nowarn NonNullResult

  /*@ non_null */  //@ nowarn NonNullResult
  Object y1() {
    return null;
  }

  /*@ non_null */ Object y2() {
    return null;
  }  //@ nowarn

  /*@ non_null */  //@ nowarn
  Object y3() {
    return null;
  }
}
