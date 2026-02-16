
class unreachable {
  void m1() {
    if (true) {
    } else {
      //@ unreachable;
    }
  }
}
