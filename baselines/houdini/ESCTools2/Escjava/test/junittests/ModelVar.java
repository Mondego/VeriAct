//#FLAGS:  -classpath . -quiet
/* FIXME
Need to test model variables of other objects.
Model variables whose definitions are recursive.
*/

public class ModelVar {


	//@ static public model int ssize;
	//@ private static represents ssize = slength+1;
	static int slength; //@ in ssize;

	//@ requires ssize == 1;
	//@ modifies ssize;
	//@ modifies slength;
	//@ ensures slength == 2;
	//@ ensures ssize == 3;
	public static void sm() {
		//@ assert ssize == 1;
		//@ assert slength == 0;
		slength++;
		//@ assert ssize == 2;
		//@ assert slength == 1;
		slength++;
	}



	//@ public model int size;
	//@ private represents size = length+1;
	int length; //@ in size;



	//@ requires size == 1;
	//@ modifies size;
	//@ modifies length;
	//@ ensures length == 2;
	//@ ensures size == 3;
	//@ ensures \old(size) == 1;
	public void m() {
		//@ assert size == 1;
		//@ assert length == 0;
		length++;
		//@ assert size == 2;
		//@ assert length == 1;
		length++;
	}
		
	//@ public non_null model int[] a;
	final public /*@ non_null */ int[] ac = new int[10]; //@ in a;
	//@ represents a = ac;
	//@ public invariant a.length > 4;

	//@ requires a[2] == 5;
	//@ modifies a[2];
	//@ ensures a[2] == 7;
	//@ ensures a.length > 5;
	public void mm() {
		
		ac[2] = ac[2] + 1;
		//@ assert a[2] == 6;
		ac[2] = ac[2] + 1;
	}
	// FIXME - need a[x] to model ac[x]?  Why may we modify ac[2]?

	//@ public model int size2;
	//@ represents size2 = size * 2;

	//@ requires size2 == 10;
	//@ modifies length, size, size2;
	//@ ensures size == 8;
	public void m2() {
		length+= 2;
		//@ assert size2 == 14;
		length += 1;
	}


	/*@ non_null */ public Other o = new Other();
	//@ public model int lenm;
	//@ represents lenm = o.mod +5;
	//@ modifies o.i, o.mod, lenm;
	//@ ensures lenm == 17;
	public void m2a() {
		o.i = 1;
		//@ assert lenm == 16;
		o.i = 2;
	}
	

	private ModelVar next;
	//@ public model int len;
	//@ represents len = next == null ? 0 : 1 + next.len;
	// @ invariant len >= 0;

/* FIXME - sends Simplify into a long think
	//@ requires len == 0;
	//@ modifies next,len;
	//@ ensures len == 1;
	public void mmm() {
		next = new ModelVar();
		// @ assert next.len == 0;
	}
*/

	//@ modifies this.*;
	//@ ensures next == null;
	//@ ensures len == 0;
	public ModelVar();
	
}

class Other {

	public int i;
	//@ public model int mod;
	//@ represents mod = i+10;

}
