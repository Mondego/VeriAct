class C {
  boolean x, y;

  /*@ requires x
          && y; */
  void m0() {
  }

  /*@ requires x && y;
        */
  void m1() {
  }

  //@ requires x && y
  void m2() {
  }

  //@ requires x && y;
  void m3() {
  }

  /*@ requires x && y */
  void m4() {
  }

  /*@ non_null
   */ int[] a = new int[25];

  /** "<esc> monitored_by this
      , this.a </esc> */
  Object b;

  /** "<esc> monitored
      </esc> */
  Object c;

  void caller0(int t) {
    switch (t) {
      case 0:  m0(); break;
      case 1:  m1(); break;
      case 2:  m2(); break;
      case 3:  m3(); break;
      case 4:  m4(); break;
      case 5:  a = null; break;
      case 6: { /*@ uninitialized
		       non_null */ int[] a = null; a = a; } break;
      case 7:  b = a; break;
      case 8:  c = a; break;
    }
  }

/*@requires s=="h</esc>\"</esc>'</esc>\n\034"+'c'+'\''+"</esc>"
        */
  void p0(String s) {
  }

  /** //   <esc> requires b != null
              </esc> */
  void p1() {
  }

  /** "//"   <esc> requires b != null
              </esc> */
  void p2() {
  }

  //@ /* h'\\\\\ ;'' a cmnt" y */ ensures \result < -1
  //@ requires true|"*/\""=="</esc>"; ensures \result > -7
  int caller1(int t) {
    switch (t) {
      case -10: case -2: case -1:  return t;
      case 0:  p0(null); break;
      case 1:  p1(); break;
      case 2:  p2(); break;
      case 3:  tz += 20; break;
      case 9:  /** <esc>unreachable
		   ; </esc>*/ break;
    }
    return -5;
  }

  int tz = 20 /* *// 2; /*@ invariant
		tz < 100; */

  //@ invariant tz>=0 /* */*0; invariant tz==10||tz%3==0

  /** <esc>ensures 0 <// comment goes here
          \result    </esc> */
  // The next test case will place the ellipsis in a slightly awkward place
  /*@ ensures c // why "c" one wonders
              != null; */
  int q(int t) {
    return t;
  }
}
