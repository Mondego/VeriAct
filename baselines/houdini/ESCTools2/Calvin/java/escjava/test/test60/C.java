/* Copyright Hewlett-Packard, 2002 */

class Super {
  //@ ensures 0 <= \result;
  public int m(int x) {
    return 0;
  }
  
  //@ requires (x & 3) == 2;
  public int p(int x) {
    return 0;
  }
}

interface J {
  //@ ensures \result != x;
  public int m(int x);
  
  //@ requires x % 2 == 0;
  public int p(int x);
}

interface K extends J {
  public int p(int x);
}

interface M extends K, J {
}

class C extends Super implements J {
  //@ also_ensures \result < 10;
  public int m(int x) {
    return x+x;
  }
  
  public int p(int x) {
    //@ assert (x & 3) == 2;
    //@ assert x % 2 == 0;
    return 0;
  }
}

class D implements M {
  //@ also_requires x < 1000;
  public int p(int x) {
    //@ assert (x & 3) == 2;
    //@ assert x % 2 == 0;
    //@ assert x < 1000;
    return x;
  }
}
