class A {
  // bug in javac, says f() is not reached
  // javafe doesn't do flow sensitive checking yet
  void f() {
  x: 
    break x;
    f();
  }
}
  
