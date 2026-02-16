class chain2 {
  void m() {
    p();
  }

  //@ helper
  private void p() {
    p();
  }
}
