/* Copyright Hewlett-Packard, 2002 */

// EscJavaTest000
// Test 0, ESC/Java test suite
// Copyright (c) 1998, Digital Equipment Corporation
// Revision history:
//   27 Feb 1998  rustan     Created

package escjava.testsuite;


public class EscJavaTest000 {
  int x;
  static int g;

  int m(int a, int b) {
    return a + b + x + g;
  }
}
