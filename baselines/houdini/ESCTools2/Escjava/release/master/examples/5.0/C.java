class C {

  int n;

  static int m(C a, C b) {
    if (a != null) {
      return a.n;
    } else {
      return b.n;
    }
  }
}
