// Tests when fresh or local items are modified
public class ModChecks6 {
	//@ ensures \fresh(a);
	//@ modifies this.*;
	ModChecks6();

	public int i;
	//@ non_null
	public int[] a = new int[10];
	//@ public invariant a.length > 6;

	//@ modifies \nothing;
	public void m() {
	    int[] aa = new int[6];
	    aa[1] = 0;
	    ModChecks6 o = new ModChecks6();
	    o.i = 0;
	    o.a[0] = 0; // WARNING
	    o.mm();
	    int j;
	    j = 0;
	}

	//@ modifies this.*;
	//@ modifies ModChecks6.*;  // WARNING
	//@ modifies a[0..1];
	//@ modifies a[0];
	//@ modifies a[*];
	public void mm();
}
	
