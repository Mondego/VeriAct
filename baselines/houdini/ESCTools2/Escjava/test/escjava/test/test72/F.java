class F {
  void bound0(int x) {  // always okay
    //@ decreases x;
    while (0 <= x) {
      x--;
    }
  }

  void bound1(int x) {  // bound error always detected
    //@ decreases x;
    while (true) {
      x--;
    }
  }

  //@ requires 0 <= x;
  void bound2(int x) {  // bound error detected with 2 or more unrollings
    //@ decreases x;
    while (true) {
      x--;
    }
  }

  //@ requires 3 <= x;
  void bound3(int x) {  // bound error detected with 5 or more unrollings
    //@ decreases x;
    while (true) {
      x--;
    }
  }

  //@ requires 4 <= x;
  void bound4(int x) {  // bound error detected with 6 or more unrollings
    //@ decreases x;
    while (true) {
      x--;
    }
  }

  void bound5(int x) {  // no bound error, unbeknownst to loopSafe
    boolean b = false;
    //@ decreases x;
    while (x != 0) {
      if (!b && x < 10) {
	break;
      }
      b = true;
      x--;
    }
  }

  void bound6(int x) {  // no bound error, and loopSafe knows it
    boolean b = false;
    //@ loop_invariant b ==> 0 <= x;
    //@ decreases x;
    while (x != 0) {
      if (!b && x < 10) {
	break;
      }
      b = true;
      x--;
    }
  }

  void decr0(int x) {  // always decreases
    //@ decreases x;
    while (0 <= x) {
      x--;
    }
  }

  void decr1(int x) {  // decrease error always detected
    //@ decreases x;
    while (0 <= x) {
      if (x != 0) {
	x--;
      }
    }
  }

  //@ requires 1 <= x;
  void decr2(int x) {  // decrease error detected with 2 or more iterations
    //@ decreases x;
    while (0 <= x) {
      if (x != 0) {
	x--;
      }
    }
  }

  //@ requires 4 <= x;
  void decr3(int x) {  // decrease error detected with 5 or more iterations
    //@ decreases x;
    while (0 <= x) {
      if (x != 0) {
	x--;
      }
    }
  }

  //@ requires 6 <= x;
  void decr4(int x) {  // decrease error detected with 6 or more iterations
    //@ decreases x;
    while (0 <= x) {
      if (x != 0) {
	x--;
      }
    }
  }

  void decr5(int x) {  // no decrease error, unbeknownst to loopSafe
    boolean b = false;
    //@ decreases x;
    while (0 <= x) {
      if (b) {
	x = 80;
	b = false;
      } else if (!b && 100 <= x) {
	b = true;
      }
      x--;
    }
  }

  void decr6(int x) {  // no decrease error and loopSafe knows it
    boolean b = false;
    //@ loop_invariant b ==> 99 <= x;
    //@ decreases x;
    while (0 <= x) {
      if (b) {
	x = 80;
	b = false;
      } else if (!b && 100 <= x) {
	b = true;
      }
      x--;
    }
  }
}
