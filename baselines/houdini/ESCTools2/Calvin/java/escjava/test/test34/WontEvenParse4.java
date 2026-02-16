/* Copyright Hewlett-Packard, 2002 */

class WontEvenParse4 {
  //@ ensures \result == (3 < 4 ? new Object() : new int[12]);
  Object m4() {
    return 3 < 4 ? new Object() : new int[12];
  }
}
