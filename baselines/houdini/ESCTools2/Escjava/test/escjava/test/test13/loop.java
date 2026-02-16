
public class loop {
  int m1_while_ok() {
    while (false) {}
  }
  int m2_while_fail() {
    while (false) {}
    //@ assert false;
  }
  int m3_while_ok() {
    while (true) {}
    //@ assert false;
  }
  
  int m1_do_ok() {
    do {} while (false);
  }
  int m2_do_fail() {
    do {} while (false);
    //@ assert false;
  }
  int m3_do_ok() {
    do {} while (true);
    //@ assert false;
  }
  
  
  int m1_for_ok() {
    for(;false;) {}
  }
  int m2_for_fail() {
    for(;false;) {}
    //@ assert false;
  }
  int m3_for_ok() {
    for(;true;) {}
    //@ assert false;
  }

  int r0_fast_finds_error() {
    int[] a = null;
    int i = 1;
    while (i <= 10) {
      a[i] = 0; // null dereference error
      i++;
    }
    return i;
  }

  int r1_fast_misses_error() {
    int[] a = new int[10];
    int i = 1;
    while (i <= 10) {
      a[i] = 0; // index out-of-bounds error
      i++;
    }
    return i;
  }

  int r2_fast_finds_error(int[] a)
  /*@ requires a != null; */
  {
    int i = 1;
    int s = 0;
    while (i <= a.length) {
      s += a[i]; // index out-of-bounds error
      i++;
    }
    return s;
  }
  
  int r3_fast_misses_error(int[] a)
  /*@ requires a != null; */
  {
    //@ assume a.length == 10;
    int i = 1;
    int s = 0;
    while (i <= a.length) {
      s += a[i]; // index out-of-bounds error
      i++;
    }
    return s;
  }
  
  // The following test case was once used to home in on the error actually
  // detected by the next case.
  int r4_checker_is_off_the_hook(int[] a)
  /*@ requires a != null; */
  {
    int i = 1;
    int s = 0;
    while (i <= a.length) {
      // the following assumption is bogus:
      //@ assume i < a.length;
      s += a[i]; // index out-of-bounds error
      i++;
    }
    return s;
  }

  // The following test case once caused an error in the checker because
  // the "length" field, when occurring in a Java expression (as opposed
  // to in a SpecExpr), was not translated into an application of the
  // "arrayLength" function.
  int r5_no_error(int[] a)
  /*@ requires a != null; */
  {
    int i = 1;
    int s = 0;
    while (i <= a.length) {
      if (i < a.length)
      {
        s += a[i];
      }
      i++;
    }
    return s;
  }

  // The following case contains no error.  It is intended to check whether
  // ESC/Java makes the "break" labels for the inner and outer loops distinct.
  void r6_no_error(int k) {
    int i = 0;
    Outer:
    while (i < 10) {
      int j = k;
      Inner:
      while (j < 10) {
        j++;
        if (j == 6)
        {
          break Outer;
	}
      }
      //@ assert j != 6;
      i++;
    }
  }

  // The following case contains no error.  It is intended to check whether
  // ESC/Java makes the "break" label for a loop distinct from "ec$throw".
  /*@ requires t != null; */
  void r7_no_error(int j, Throwable t) {
    try {
      while (j < 10) {
        j++;
        if (j == 6)
        {
          throw t;
	}
      }
      //@ assert j != 6;
    }
    catch (Throwable tt) {
    }
  }

  // The following case contains no error.  It is intended to check whether
  // ESC/Java makes the "break" label for a loop distinct from "ec$return".
  void r8_no_error(int j) {
    while (j < 10) {
      j++;
      if (j == 6)
      {
        return;
      }
    }
    //@ assert j != 6;
  }

  void r9_fast_misses_error() {
    int j = 0;
    while (j < 10) {
      // loop_invariant j <= 10;
      j++;
    }
    // The error is that the following line really *is* reachable.
    //@ unreachable;
  }
}

class loopinvariant extends loop {
  int j0_invariant() {
    int x = 0;
    /*@ loop_invariant x <= 10; */
    while (x < 10) {
      x += 2;
    }
    return x;
  }

  int j1_invariant() {
    int x = 0;
    /* loop_invariant \old(x) <= x && x <= 10 */
    while (x < 10) {
      x += 2;
    }
    // assert \old(x) == 5;
    return x;
  }
}
