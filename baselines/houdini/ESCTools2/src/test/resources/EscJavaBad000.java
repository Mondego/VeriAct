// EscJavaBad001
// Bad test 0, ESC/Java test suite
// Copyright (c) 1998, Digital Equipment Corporation
// Revision history:
//   27 Feb 1998  rustan     Created

// This test contains an integer division operation that may fail
// with a division-by-zero error.

// A positive version of this test is found in TestJavaTest001.



public class EscJavaBad000 {
  int m(int x, int y) {
    return x / y;  // potential division-by-zero error
  }
}
