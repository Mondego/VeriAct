public class OldFormals {

  public int[] a = new int[10];

  int k;
  //@ invariant a != null;

  //@ requires 0 < i && i < a.length;
  //@ modifies a[i-1];
  public void m(int i) {

	a[--i] = 0;
	//@ assert a[\old(i-1)] == 0;
  }

  //@ requires j == 5;
  //@ modifies k;
  //@ ensures j == 5;
  public void n(int j) {
	--j;
	//@ assert j == 4;
	//@ assert \old(j) == 5;
   }

   //@ requires i == 4;
   public void q(int i) {
      i = 5;
      //@ ghost int g = \old(i);
      //@ assert g == 4;
   }

   public void r(int i) {
	i = i+1;
	//@ assume \old(i) == 4;
	//@ assert i == 5;
   }

   //@ requires i == 4;
   public void t(int i) {
	i = 5;
	//@ ghost int g = 6;
        //@ set g = \old(i);
	//@ assert i == g+1;
   }
}
