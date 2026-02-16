public class Inconsistent {
  public void m() {
    int a,b,c,d;
    //@ assume a == b;
    //@ assume b == c;
    //@ assume a != c;
    //@ assert a == d; // Passes, but inconsistent
    //@ assert false;  // Passes, but inconsistent
  }
}
