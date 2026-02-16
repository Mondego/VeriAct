public class StoreRefs extends StoreRefsB {

	static public Object o;
	public int i;
	public int[] a;
	public StoreRefsA sr;
	public StoreRefsA[] s;

/*
	//@ modifies i, a[0], o;
	//@ modifies StoreRefs.o;
	//@ modifies StoreRefsA.srf.a[0];
	//@ modifies this.i, this.a[0];
	//@ modifies s[0].z, s[0].aa[3];
	//@ modifies s[0 .. 4];
	//@ modifies s[0 ..4];
	//@ modifies s[0..4];
	//@ modifies s[*], s[0].aa[*];
	//@ modifies s[0..4].z;
	//@ modifies s[0..4].ss[*];
	//@ modifies this.*, StoreRefs.*, s[0].*;
	//@ modifies super.i;
*/
	//@ modifies StoreRefs.o;
	//@ modifies StoreRefs.*;
	//@ modifies java.lang.Object.*;
	void m() {}
}

class StoreRefsA {

	public int z;
	public int[] aa;
	public StoreRefsA[] ss;
	static public StoreRefs srf;
}

class StoreRefsB {

	public int i;
}
