class D {
  static int x;

  
  /** Both of the following sets of pragmas are supposed to
    * be recognized by ESC/Java:
    *     <esc>ensures \result < 0;</esc>
    * and:
    *     <esc>modifies x; ensures x < 0;</esc>
    */
  static int p(int s, int t) {
    x = s;
    return t;
  }
}
