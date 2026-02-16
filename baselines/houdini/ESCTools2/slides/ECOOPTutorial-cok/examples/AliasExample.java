// This example shows the use of owner fields to control reasoning 
// problems with aliasing

public class AliasExample {

  boolean allNull = true;
  /*@ non_null */ Object[] a = new Object[2];
  //@ invariant allNull == (a[0]==null && a[1]==null);

  //@ requires i==0 || i==1;
  public void insert(int i, Object o) {
    a[i] = o;
    if (o != null) allNull = false;
    if (o == null) allNull = (a[1-i]==null);
  }
}
    
