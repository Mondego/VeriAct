class BogusModifiers {
  void good(int t) {
    final int x = 3;
    for (final int i = 0; i < 100; ) {
    }
    for (/*@ non_null */ BogusModifiers nf = null; t < 10; t++) {
    }
    for (final /*@ non_null */ BogusModifiers nf = null; t < 20; t++) {
    }
    for (/*@ non_null */ final BogusModifiers nf = null; t < 30; t++) {
    }
    for (/*@ readable_if 30 <= t; */ BogusModifiers nf = null; t < 30; t++) {
      Object o = nf;
    }
    for (/*@ readable_if 30 <= t; */ BogusModifiers nf = null; t < 30; t++) {
      Object o = nf;
    }
    for (BogusModifiers nf = null, xf = null /*@ readable_if 30 <= t; */; t < 30; t++) {
      Object o = nf;
    }
  }
  
  abstract native int xx = 0;

  void m0() {
    public int x0 = 0;
    private int x1 = 0;
    protected int x2 = 0;
    static int x3 = 0;
    final int x4 = 0;  // this one is fine for local variables
    // Although "synchronized" is a modifier, the Javafe parser does not expect
    // it to be a modifier in this context, so it tries to parse the following
    // as a statement, which generates a parsing error:
    // synchronized int x5 = 0;
    volatile int x6 = 0;
    transient int x7 = 0;
    native int x8 = 0;
    abstract int x9 = 0;
  }
  
  void m1() {
    /*@ spec_public */ int x0 = 0;
    /*@ non_null */ int x1 = 0;
    /*@ requires true; */ int x2 = 0;
    /*@ modifies x0 */ int x3 = 0;
    /*@ ensures true; */ int x4 = 0;
    /*@ exsures (Throwable t) true; */ int x5 = 0;
    int x6 = 0 /*@ exsures (Throwable t) true */;
  }
}
