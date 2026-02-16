/* Copyright Hewlett-Packard, 2002 */

/**
 ** This file tests parsing of escjava annotations in javadoc comments
 **/

class JavaDocCheck {

  /** <esc>requires x == 0;</esc><esc>requires y == 1;   </esc>
    *
    *
    *  <esc>
    *       requires z == 2;
    *  </esc>
    **/
  /** <esc>requires w
    * == 3;</esc> */

  // The following is not a Javadoc comment, so ESC/Java ignores what's
  // inside of it.
  //** <esc>ensures \result == 8;</esc>
  int m(int x, int y, int z, int w) {
    //@ assert x == 0;
    //@ assert y == 1;
    //@ assert z == 2;
    //@ assert w == 3;
    return 10;
  }

  int n() {
    // The following produces a warning, just to see that the file really
    // is being checked.
    //@ unreachable
  }
}
