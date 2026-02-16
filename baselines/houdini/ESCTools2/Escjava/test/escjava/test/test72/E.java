class E {
  // This file contains some tests regarding the details of our
  // handling of 'decreases'.

  void m0(int x) {
    // The following is fine, because of the way we check the lower bound.
    // Note that x may be negative on entry to the loop, but only it would
    // have been 0 just before the loop was evaluated.
    //@ decreases x;
    while (x-- >= 0) { }
    //@ assert x < -1;
  }
  void m1(int x) {
    // The following is also fine.  Here, x is always negative on exit
    // from the loop.
    //@ decreases x;
    while (x-- > 0) { }
    //@ assert x < 0;
  }
  void m2(int x) {
    //@ decreases x-1;
    while (x-- >= 0) { }  // variant negative
  }
  void m3(int x) {
    //@ decreases x-1;
    while (x-- > 0) { }   // variant negative
  }
  void m4(int x) {
    int tmp;
    //@ decreases x;
    while ((tmp = x) == (x = 1000) || true) {
      x = tmp-1;
    }
  }
  void m5(int x) {
    int tmp;
    //@ decreases x;
    while ((tmp = x) == (x = 1000) || true) {
      x--;
    }
  }

  void p0(int x) {
    //@ decreases x;
    while (x > 0) {
      x--;
    }
  }
  void p1(int x) {
    //@ decreases x;
    while (true) {
      if (! (x > 0)) {
	break;
      }
      x--;
    }
  }
  void p2(int x) {
    //@ decreases j;
    for (int j = x; j > 0; j--) {
    }
  }
  void p3(int x) {
    //@ decreases x;
    do {
    } while (x-- >= 0);
  }
  void p4(int x) {
    //@ decreases x;
    while (true) { }
  }
  void p5(int x) {
    //@ decreases x;
    for ( ; ; ) { }
  }
  void p6(int x) {
    //@ decreases x;
    do { } while (true);
  }

  void q0(int x) {
    //@ decreases x;
    while (0 <= x) {  // does not necessarily decrease variant function
      if (x % 13 < 7) {
	x--;
      } else if (x / 34 == 5) {
	x += 2;
      }
      x--;
    }
  }
  void q1(int x) {
    //@ decreases x;   // not decreased, and
    while (-2 <= x) {  // negative variant function does not lead to exit
      if (x == -2) {
	break;
      }
    }
  }
  void q2(int x) {
    //@ decreases x;
    while (-2 <= x) {  // does not decrease variant function
      if (x == -2) {
	break;
      }
      x++;
      if (x == 0) {
	break;
      }
    }
  }
  void q3(int x) {
    //@ decreases x;
    while (-2 <= x) {
      if (x == -2) {
	break;
      }
      x -= 5;
      if (x == -6) {
	break;
      }
    }
  }

  //@ requires 95 <= z;
  void r0(int x, int z, boolean b) {
    //@ decreases t;
    //@ loop_invariant 95 <= t;
    for (int t = z; t != 95; t--) {
      r1(x);
      r1(x+1);
      //@ assert b;  // just to check we get here
    }
  }
  //@ helper
  final void r1(int y) {
    //@ decreases y;
    while (y > 0) {
      y--;
    }
  }

  void s0(int x, int y) {
    //@ decreases x;
    //@ decreases y;
    while (true) {
      if (x < 0 || y < 0) {
	break;
      }
      x--;
      y--;
    }
  }
}
