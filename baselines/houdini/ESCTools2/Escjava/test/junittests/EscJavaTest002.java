// EscJavaTest002
// Test 2, ESC/Java test suite
// Copyright (c) 1998, Digital Equipment Corporation
// Revision history:
//   27 Feb 1998  rustan     Created

// This test includes a simple object invariant that is established
// by a member initializer and maintained by the class's single method.


public class EscJavaTest002 {
  int n = 0;
  /*@ invariant 0 <= n; */

  void inc() {
    n++;
  }
}
