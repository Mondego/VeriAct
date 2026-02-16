class B {
  // bug in javac, causes compiler crash
  // ok in javafe
  void g() {
  x:
    continue x;
    g();
  }
}
