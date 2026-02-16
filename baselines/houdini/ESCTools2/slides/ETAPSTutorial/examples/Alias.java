public class Alias {
  /*@ non_null */ int[] a = new int[10];
  boolean noneg = true;

  /*@ invariant noneg ==> 
                 (\forall int i; 0<=i && i < a.length;  a[i]>=0); */

  //@ requires 0<=i && i < a.length;
  public void insert(int i, int v) {
    a[i] = v;
    if (v < 0) noneg = false;
  }

/*
  Alias() {
	//@ set a.owner = this;
  }
  //@ invariant a.owner == this;
*/
}

