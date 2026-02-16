class E {
  boolean x, y, z;

  //@ requires x ==> y ==> z;
  void m0() {
    if (x) {
      if (y) {
	//@ assert z;
      } else {
	//@ assert z;  // should generate warning
      }
    } else {
      //@ assert y;  // should generate warning
      //@ assert z;  // should generate warning
    }
  }

  //@ requires (x ==> y) ==> z;
  void m1(boolean t) {
    if (x) {
      if (y) {
	//@ assert z;
      } else {
	//@ assert z;  // should generate warning
      }
    } else if (t) {
      //@ assert y;  // should generate warning
    } else {
      //@ assert z;
    }
  }

  //@ requires z <== y <== x;
  void m2() {
    if (x) {
      if (y) {
	//@ assert z;
      } else {
	//@ assert z;  // should generate warning
      }
    } else {
      //@ assert y;  // should generate warning
      //@ assert z;  // should generate warning
    }
  }

  //@ requires z <== (y <== x);
  void m3(boolean t) {
    if (x) {
      if (y) {
	//@ assert z;
      } else {
	//@ assert z;  // should generate warning
      }
    } else if (t) {
      //@ assert y;  // should generate warning
    } else {
      //@ assert z;
    }
  }

  void p(boolean t) {
    if (t) {
      //@ assert x <==> y <==> z <==> t <==> z <==> y <==> x;
      //@ assert x <=!=> y <=!=> z <=!=> t <=!=> z <=!=> y <=!=> x;
    } else {
      //@ assert x <==> y <=!=> z <==> t <==> z <==> y <==> x;
      //@ assert x <=!=> y <=!=> z <=!=> t <=!=> z <==> y <=!=> x;
    }
  }

  void q() {
    // the following should fail to verify:
    //@ assert y <=!=> x <==> y <==> x;
  }
}
