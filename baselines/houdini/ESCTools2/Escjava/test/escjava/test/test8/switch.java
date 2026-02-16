class Switch {
  //@ requires p == 3;
  void m0(int p) {
    int x=0;
    switch(p) {
      case 2:
	x=2;
	break;
      case 3:
	x=3;
	break;
      default:
	x=5;
    }
    //@ assert x==3;
  }

  void m1() {
    int x=0;
    switch(2+1) {
      case 2:
	x=2;
	break;
      case 3:
	x=3;
	break;
      default:
	x=5;
    }
    //@ assert x==3;
  }

  void m2() {
    int x=0;
    switch(5) {
      case 2:
	x=2;
	break;
      default:
	x=5;
    }
    //@ assert x==5;
  }

  void m3() {
    int x=0;
    switch(6) {
      case 2:
	x=2;
	break;
    }
    //@ assert x==0;
  }

  void m4() {
    int x=0;
    switch(6) {
      case 2:
	x=2;
	break;
    }
    // The following should fail:
    //@ assert x==2;
  }

  void m5(int y) {
    int x=0;
    switch(y) {
      case 2:
	x=2;
	break;
    }
    // The following should fail:
    //@ assert x==0;
  }

  void m6(int y) {
    int x=0;
    switch(y) {
      case 2:
	x=2;
	break;
    }
    // The following should fail:
    //@ assert x==2;
  }
}

