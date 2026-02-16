class C {
  int i;

  boolean m(boolean i) {
    return i;   // should not get field i
  }
}
