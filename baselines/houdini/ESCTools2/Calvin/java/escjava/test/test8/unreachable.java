/* Copyright Hewlett-Packard, 2002 */


class unreachable {
  void m1() {
    if (true) {
    } else {
      //@ unreachable;
    }
  }
}
