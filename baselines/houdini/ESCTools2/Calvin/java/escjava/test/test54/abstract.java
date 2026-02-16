/* Copyright Hewlett-Packard, 2002 */

abstract class T {
  /*@ helper */ abstract void m();
  /*@ helper */ native final void n();
}

interface J {
  /*@ helper */ void p();
}
