class C {
  void m0(int x) {
    //@ decreases 10-x;
    while (x < 10) {
      x++;
    }
  }

  void m1(int x) {
    //@ decreases 10-x;
    do {
      x++;
    } while (x < 10);
  }

  void m2(int t) {
    //@ decreases 10-x;
    for (int x = t; x < 10; x++) {
    }
  }

  //@ requires x <= 10;
  void m3(int x) {
    //@ decreases 10-x;
    //@ loop_invariant x <= 10;
    //@ decreases 100-x;
    //@ decreases 84-x;
    while (x < 10) {
      x++;
    }
  }

  void p0(int x) {
    //@ decreases 9-x;
    while (x < 10) {
      x++;
    }
  }

  void p1(int x) {
    //@ decreases 9-x;
    do {
      x++;
    } while (x < 10);
  }

  void p2(int t) {
    //@ decreases 9-x;
    for (int x = t; x < 10; x++) {
    }
  }

  //@ requires x <= 10;
  void p3(int x) {
    //@ loop_invariant x <= 10;
    //@ decreases 99-x;
    //@ decreases 9-x;
    //@ decreases 83-x;
    while (x < 10) {
      x++;
    }
  }

  void q0(int x) {
    //@ decreases 8-x;  // variant negative
    while (x < 10) {
      x++;
    }
  }

  void q1(int x) {
    //@ decreases 8-x;
    do {
      x++;
    } while (x < 10);
  }

  void q2(int t) {
    //@ decreases 8-x;  // variant negative
    for (int x = t; x < 10; x++) {
    }
  }

  //@ requires x <= 10;
  void q3(int x) {
    //@ loop_invariant x <= 10;
    //@ decreases 98-x;
    //@ decreases 8-x;  // variant negative
    //@ decreases 82-x;
    while (x < 10) {
      x++;
    }
  }

  void r0(int x) {
    //@ decreases 7-x;  // variant negative
    while (x < 10) {
      x++;
    }
  }

  void r1(int x) {
    //@ decreases 7-x;  // variant negative
    do {
      x++;
    } while (x < 10);
  }

  void r2(int t) {
    //@ decreases 7-x;  // variant negative
    for (int x = t; x < 10; x++) {
    }
  }

  //@ requires x <= 10;
  void r3(int x) {
    //@ loop_invariant x <= 10;
    //@ decreases 97-x;
    //@ decreases 7-x;  // variant negative
    //@ decreases 81-x;
    while (x < 10) {
      x++;
    }
  }
}
