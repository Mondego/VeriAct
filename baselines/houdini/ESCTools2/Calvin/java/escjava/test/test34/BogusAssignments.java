/* Copyright Hewlett-Packard, 2002 */

class BogusAssignments {
  final int x = 5;
  void m() {
    x = 3;  // this is a "final" field, but we don't currently disallow them
  }
  
  void n() {
    int[] a = null;
    a.length = 3;
  }
  
  //@ modifies a.length;
  //@ modifies x;
  void p(int[] a) {
  }
  
  void q() {
    int[] a = null;
    p(a);
  }
}
