class I {
  // bug in javac, switch body should start with switch label
  // ok in javafe
  void n() {
    switch(0) {
      int i=0;
    case 0:
      i=1;
    }
  }
}
