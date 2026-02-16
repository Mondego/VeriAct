class D extends B {
  private C a;
  //@ represents isInit = (a!=null && a.isInit);

  public int m() {
    return a.m();
  }
}
