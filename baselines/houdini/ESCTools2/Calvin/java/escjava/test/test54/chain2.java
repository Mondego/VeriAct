/* Copyright Hewlett-Packard, 2002 */

class chain2 {
  void m() {
    p();
  }

  //@ helper
  private void p() {
    p();
  }
}
