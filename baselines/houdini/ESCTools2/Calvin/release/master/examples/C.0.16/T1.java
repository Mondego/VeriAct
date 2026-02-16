/* Copyright Hewlett-Packard, 2002 */

class T1 {
  int x;

  //@ requires b;
  //@ ensures this.x == 0;
  T1() {
    x = 0;
  }

  //@ requires b;
  //@ requires (\forall T1 t; t.x != 0);
  //@ ensures \result != null && \result.x == 0;
  //@ ensures \fresh(\result); // line (a)
  static T1 m() {
    return new T1();
  }

  //@ requires b;
  //@ requires (\forall T1 t; t.x != 0);
  static void p() {
    T1 t = m();
    int i = 1/t.x;      // Warning here depends on line (a) above
  }

  //@ requires \nonnullelements(args);
  //@ modifies b;
  public static void main(String[] args) {
    if (!b) {
      b = true;
      p();
    }
  }

  static boolean b = false;
  //@ invariant !b ==> (x == x+1);
}

