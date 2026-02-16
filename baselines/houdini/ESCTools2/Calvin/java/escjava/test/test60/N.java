/* Copyright Hewlett-Packard, 2002 */

class Super {
  public void m0(/*@ non_null */ int[] x) {
  }
  public void m1(int[] x, int[] y) {
  }
  public void m2(/*@ non_null */ int[] x) {
  }
  public void m3(/*@ non_null */ int[] x) {
  }
}

interface J {
  public void m0(int[] x);
  public void m1(/*@ non_null */ int[] x, int[] y);
  public void m2(/*@ non_null */ int[] x);
  public void m4(/*@ non_null */ int[] x);
}

class N extends Super implements J {
  public void m0(int[] x) {
    //@ assert x != null;
  }
  public void m1(int[] x, int[] y) {
    //@ assert x != null;
    //@ assert y != null;
  }
  public void m2(int[] x) {
    //@ assert x != null;
  }
  public void m3(int[] x) {
    //@ assert x != null;
  }
  public void m4(int[] x) {
    //@ assert x != null;
  }

  public void p(int which, int[] x) {
    switch (which){
      case 0:  m0(x);  break;
      case 1:  m1(x, x);  break;
      case 2:  m2(x);  break;
      case 3:  m3(x);  break;
      case 4:  m4(x);  break;
    }
  }
}
