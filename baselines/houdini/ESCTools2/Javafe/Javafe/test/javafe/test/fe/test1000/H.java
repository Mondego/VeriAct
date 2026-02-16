class H {
  // bug, shouldn't be able to continue switch
  void m() {
  x:
    switch(0) {
    case 0:
      continue x;
    }
  }
}
