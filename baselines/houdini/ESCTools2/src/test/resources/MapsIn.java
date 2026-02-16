public class MapsIn extends MapsInS {

	//@ model JMLDataGroup j,k;

	public int i;
	//@ in j,k;
	//@ in super.i, super.j;

	public int[] a;
	//@ maps a[0] \into j,k;
	//@ maps a[1 .. 2] \into j,k;
	//@ maps a[*] \into j,k;

	public MapsT b;
	//@ maps b.f \into k;
	//@ maps b.f \into super.i, super.j;

	public MapsT[] c;
	//@ maps c[0].f \into k;

	public int ii;

	//@ ghost int[] gi; in j; maps gi[0] \into k;
	//@ model int[] mi; in j; maps mi[0] \into k;

	// ERRORS

	public int z;
	//@ in
	//@ in ;
	//@ in i j
	//@ in i,,
	//@ in ,i;

	//@ maps i;
	//@ maps i j 
	//@
}

class MapsInS {
	public int i;
}

class JMLDataGroup {}

class MapsT {
	public int f;
}
