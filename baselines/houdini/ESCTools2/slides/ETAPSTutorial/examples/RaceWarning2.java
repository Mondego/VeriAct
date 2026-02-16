public class RaceWarning2 {
  Object o = new Object();
  Object oo = new Object();
  int i;
  //@ monitors_for i = o,oo;

  void m() {
    o = new Object();
    synchronized (o) { int j = i; }
    oo = null;
    o = null;
    //{ { i = 2; } }
    int k = i;
  }
}
