public class Invariant {
  public int i,j;
  //@ invariant i > 0;
  //@ constraint j > \old(j);

  public void m() {
    i = -1; // will provoke an Invariant error
    j = j-1; // will provoke a Constraint error
  }
}
