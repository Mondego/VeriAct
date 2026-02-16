class causeErrors {

  // An abstract method cannot include a body
  abstract void m1() { }

  // A native method cannot include a body
  native void m2() { }

  // Method must include a body unless it is declared abstract or native
  void m3();

  void m0() {

    // Argument of increment/decrement operation is not an lvalue
    int l = (i+1)++;

  }    
}
