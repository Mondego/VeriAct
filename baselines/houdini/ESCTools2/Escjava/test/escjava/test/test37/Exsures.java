class Exsures {
  //@ ensures false
  void m0() throws Throwable {
    throw new Throwable();
  }

  //@ ensures false
  void m1() {
    throw new IndexOutOfBoundsException();  // Java is happy with this, but
                                            // ESC/Java issues a complaint
  }

  //@ ensures false
  void m2() throws IndexOutOfBoundsException {
    throw new IndexOutOfBoundsException();  // both Java and ESC/Java are happy
                                            // with this
  }

  //@ requires 0 <= x;
  //@ ensures x == 0;
  //@ exsures (Throwable t) 0 < x;
  //@ exsures (SomeException se) 10 < x;
  //@ exsures (SomeException se) x < se.f;
  void m3(int x) throws Throwable {
    switch (x) {
      case 1:
	throw new Throwable();
      case 12:
	throw new SomeException(x+2);
      case 0:
	break;
      default:
	if (x > 20) {
	  throw new SomeException(x+10);
	} else if (0 <= x) {
	  throw new Throwable();
	}
	//@ unreachable
    }
  }

  int ff;
  
  //@ modifies ff;
  //@ exsures (SomeException se) se != p;
  //@ exsures (Throwable t) \old(ff) < ff;
  void m4(SomeException p) throws SomeException {
    if (p != null) {
      ff++;
      throw new SomeException(2);
    }
  }

  //@ requires k % m == 0;
  //@ modifies ff;
  //@ ensures k == 0;
  //@ ensures ff == 5;
  //@ exsures (SomeException ss) m != 2;
  void m5(int k, int m) throws SomeException {
    int w = 2;
    try {
      SomeException se = new SomeException(k, m);
      w = 3;
    } catch (SomeException s) {
      //@ assert m != 2;
      throw s;
    } catch (Throwable t) {
      //@ assert k == 0;
      return;
    } finally {
      ff = 5;
      // return if a "Throwable" was caught; otherwise, fall through
    }
    //@ assert w == 3;
    do {
      // skip
    } while (k != 0);
  }

  //@ modifies ff;
  //@ ensures ff == 0
  //@ exsures (SomeException se) se.f <= \old(ff);
  void m6() throws SomeException {
    if (ff == 0) {
      // skip
    } else if (10 < ff) {
      ff = -2;
      throw new SomeException(2);
    } else {
      // the following "throw" may fail to satisfy the exceptional
      // postcondition
      throw new SomeException(4);
    }
  }

  //@ modifies ff;
  //@ ensures ff == 0
  //@ exsures (SomeException se) se.f <= \old(ff);
  void m7() throws Throwable {
    if (ff == 0) {
      // skip
    } else if (10 < ff) {
      ff = -2;
      throw new SomeException(2);
    } else {
      ff = 12;
      throw new Throwable();
    }
  }

  Exsures() throws Throwable {
  }

  //@ exsures (Throwable t) super0 == super1;
  void m8(int super0, int super1) throws Throwable {
    if (super0 == super1) {
      throw new Throwable();
    }
  }

  //@ exsures (Throwable t) super0 == super1;
  void m9(int super0, int super1) throws Throwable {
    m8(super0, super1);
  }
}

class SomeException extends Throwable {
  int f;
  
  //@ ensures f == y;
  SomeException(int y) {
    f = y;
  }

  //@ ensures f == a + b;
  //@ exsures (Throwable t) a == 0 || t instanceof SomeException;
  //@ exsures (SomeException se) a % 2 != 0;
  SomeException(int a, int b) throws Throwable {
    if (a % 2 == 0) {
      f = a + b;
    } else {
      throw this;  // this seems like a bad idea, but ESC/Java won't catch it
    }
  }
}

class AlsoExsures extends Exsures {
  int gg;
  //@ also
  //@ modifies gg;
  //@ ensures ff < 10 && gg == 4;
  //@ exsures (SomeException se) gg == se.f;
  void m6() throws SomeException {
    gg = 3;
    try {
      super.m6();
    } catch (SomeException se) {
      gg = se.f - 1;
      try {
	throw se;
      } finally {
	gg++;
      }
    }
    //@ assert gg == 3;
    ++gg;
  }

  // The following constructor may fail to establish its exceptional
  // postcondition, because the implicit call to the superclass constructor
  // may throw a "SomeException".
  //@ requires ae != null;
  //@ exsures (SomeException se) ae.gg == 17 && ae.gg == se.f;
  AlsoExsures(AlsoExsures ae) throws Throwable {
    if (ae.gg == 17) {
      throw new SomeException(ae.gg);
    }
  }
  
  //@ also
  //@ exsures (Throwable t) 0 < sub0;
  //@ exsures (Throwable t) sub1 < 10;
  void m8(int sub0, int sub1) throws Throwable {
    if (0 < sub0 && sub1 < 10 && sub0 == sub1) {
      throw new Throwable();
    }
  }

  // The following method may fail to establish the "exsures" clause declared
  // in the superclass.
  //@ also
  //@ exsures (Throwable t) 0 < sub0;
  //@ exsures (Throwable t) sub1 < 10;
  void m9(int sub0, int sub1) throws Throwable {
    if (0 < sub0 && sub1 < 10) {
      throw new Throwable();
    }
  }
}
