public class GhostLocalScope {

	//@ ghost public int i;
	public int ii;
	public Object z;

	public void m() {

		//@ set i = 0;
		int kk = 0;
		/*@ non_null */ Object o = null;
		//@ ghost non_null Object ooo;
		//@ non_null ghost Object o4;
		//@ final non_null ghost Object o5;
		//@ ghost int k = 9;
		//@ set i = 1;
		//@ set k = kk;
		//@ set k = i;
		//@ set kk = 8; // ERROR
		kk = 9;
		kk = i; // ERROR
		kk = k; // ERROR
		kk = ii;

		//@ ghost int z;
		Object op = z; // OK - should be the field z

		Object y;
		//@ ghost Object y; // ERROR - Duplicate

	}

	static public void mm() {
		int kk = 0;
		kk = ii; // ERROR
		kk = i;  // ERROR
		//@ ghost int k = 0;
		//@ set k = i; // ERROR
		//@ set k = ii; // ERROR
		//@ set k = kk; // OK
		kk = k; // ERROR
	}
}

// Checks that all of the variables in a declaration end up properly declared as ghost
// variables
class GhostDup {

	//@ ghost public int a = 9, b, c = a+b;

	void m() {
		//@ ghost int x = a+b+c, y, z = a+b+c+x+y;

		//@ set a = b+c+x+y+z;
	}
}

class GhostTemp {
	//@ ghost public int a;
	public int b;
}

class GhostArray {
	//@ ghost public int[] a = new int[10];
	public int[] b = new int[10];
	//@ ghost public GhostTemp gt = new GhostTemp();
	public GhostTemp gta = new GhostTemp();

	public void m() {
		//@ set x = 0; // ERROR - nonexistent variable
		//@ set a[0] = 0;
		//@ set b[0] = 0; // ERROR
		b[0] = a[0]; // ERROR
		//@ set gt.a = 0; // OK
		//@ set gt.b = 0; // OK
		//@ set gta.a = 0; // OK
		//@ set gta.b = 0; // ERROR
		gt.a = 0; // ERROR
		gt.b = 0; // ERROR
		gta.a = 0; // ERROR
		gta.b = 0; // OK
	}
}
