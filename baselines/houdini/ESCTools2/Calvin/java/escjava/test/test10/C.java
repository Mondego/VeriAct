/* Copyright Hewlett-Packard, 2002 */

class C {

  public static void main(String[] args) {

    int[] a = { 1 };
    sort2(a);
    // System.out.println("Min is "+a[0]);
  }
  
  static void sort2(int[] a)
    {
      //@ assume a != null && a.length == 2;
      
      if( a[0] > a[1] ) {
	int t = a[0];
	a[0] = a[2];
	a[1] = t;
      }

      //@ assert a[0] <= a[1];
    }

  static void sort2again(int[] a)
    {
      //@ assume a != null && a.length == 3;
      
      if( a[0] > a[1] ) {
	int t = a[0];
	a[0] = a[2];
	a[1] = t;
      }

      //@ assert a[0] <= a[1];       // fails because a[2] should be a[1]
    }
}
