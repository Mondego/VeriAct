// Tests the modifies \everything clause

public class ModEverything extends ModEverythingS {

	public ModEverything();

	//@ non_null
	public ModEverythingA a,b;

	public static int i;
	public int j,k,p,q;
	//@ non_null
	public int[] ar; //@ nowarn IndexTooBig;

	//@ public invariant j > 0; // FAILS

	//@ requires arg == 101;
	//@ ensures i == 11;  // OK
	//@ ensures j == 15;  // FAILS
	//@ ensures k == 8;   // FAILS
	//@ ensures p == 25;   // FAILS
	//@ ensures ar[5] == 0; // FAILS
	//@ ensures ar[6] == 0; // OK
	//@ ensures super.f == 31; // FAILS
	//@ ensures super.sf == 31; // FAILS
	//@ ensures g == 32; // OK
	//@ ensures sg == 32; // OK
	//@ ensures q == 101; // OK
	//@ ensures b.z == 45; // OK
	//@ ensures a.sz == 46; // OK
	//@ ensures a.z == 47; // FAILS
	//@ ensures a.ssz == 48; // FAILS
	//@ ensures b == \old(b); // FAILS
	//@ ensures b.zz == \old(b.zz); // OK

	public void m(int arg) {

		i = i+1;
		j = 15;
		k = 7;
		ar[5] = 0; //@ nowarn IndexTooBig;
		a.z = 47;
		a.ssz = 48;
		p = 25;
		super.f = 31;
		super.sf = 31;


		mm();
		i = i+1+i;
		k = k + 1;
		super.g = 32;
		super.sg = 32;
		q = arg;
		b.z = 45;
		b.sz = 46;
	}

	//@ modifies \everything;
	//@ ensures i == 5;
	//@ ensures ar[6] == 0;
	//@ ensures a == \old(a);
	//@ ensures b.zz == \old(b.zz);
	public void mm();
}

class ModEverythingS {

	public int f,g;
	static public int sf,sg;
}

class ModEverythingA {

	public static int sz,ssz;
	public int z,zz;

	//@ non_null
	public ModEverythingA a = this;
}
