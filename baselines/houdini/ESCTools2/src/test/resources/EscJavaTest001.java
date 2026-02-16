// EscJavaTest001
// Test 1, ESC/Java test suite
// Copyright (c) 1998, Digital Equipment Corporation
// Revision history:
//   27 Feb 1998  rustan     Created

// This test contains an integer division operation that succeeds
// because of the surrounding if-statement guard.

// A negative version of this test is found in TestJavaBad000.


public class EscJavaTest001 {
  int m(int x, int y) {
    if (y == 0) {
      return 0;
    } else {
      return x / y;
    }
  }
}
