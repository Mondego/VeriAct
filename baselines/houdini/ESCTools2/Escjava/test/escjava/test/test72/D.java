class D {
  void p0() {
    int x = 5;
    while (true) {
      //@ decreases x;   // wrong placement
    }
  }

  void m0() {
    //@ decreases x;   // undefined variable
    while (true) {
      int x = 5;
    }
  }

  void m1() {
    //@ decreases x;   // undefined variable
    do {
      int x = 5;
    } while (true);
  }

  void m2() {
    //@ decreases x;   // undefined variable
    for (int t = 0; t < 10; t++) {
      int x = t;
    }
  }
}
