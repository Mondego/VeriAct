/* In this test the assert on Line A appropriately fails if LineB is 
 not commented.  However, if Line B is commented and Line C is not
commented, then Line A does not fail.  This seems an obvious expectation
of the assume/assert mechanism.
*/

public class Arrays {
	public Arrays(){}

	public static final int[] aa = new int[10];

	//@ requires aa != null && aa.length >=10;
	public void mmm() {
		int[] a = new int[10];
		//for (int p=0; p<9; ++p) a[p] = p;
		//a[5] = 5; // Line B
		// assume (\forall int q; 0<=q && q <9; a[q] == q);
		// assume a[0]==0 && a[1]==1 && a[2]==2 && a[5]==5;
		//@ assume a[5] == 5; // Line C
		//@ assert a[5] == 6;  // ERROR // Line A
		// assert a[9] == 10; // ERROR
	}
}
