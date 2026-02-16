/* Copyright Hewlett-Packard, 2002 */

class T {
  int x;

  //@ requires b;
  //@ ensures this.x == 0;
  T() {
    x = 0;
  }

  //@ requires b;
  //@ requires (\forall T t; t.x != 0);
  //@ ensures \result != null && \result.x == 0;
  static T m() {
    return new T();
  }

  //@ requires b;
  //@ requires (\forall T t; t.x != 0);
  static void p() {
    T t = m();
    int i = 1/t.x;  // will divide by 0, but no warning
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

