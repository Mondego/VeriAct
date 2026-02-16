

public class ModChecksArgs {

	ModChecksArgs();

	public int[] a = new int[10];
	//@ invariant a != null && a.length > 5;

	//@ modifies a[0];
	public void m() {
	    ma(0,a);
	    ma(1,a);
	}

	//@ requires aa != null;
	//@ requires i >= 0 && i < aa.length;
	//@ modifies aa[i];
	public void ma(int i, int[] aa) ;

}
