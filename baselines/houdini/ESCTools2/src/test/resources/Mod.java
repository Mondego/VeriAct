public class Mod {


	//@ non_null
	public static int[] ar;

	//@ public invariant ar.length > 10;

	public int[] fr;

	public static int i;

	public int j;

	public ModA a,b;

	//@ ensures \result == \old(i) + 1;
	//@ ensures \old(j + i + ar[i] + fr[i] + a.jj + ModA.ii) * 0 == 0;
	public int mm() {
		i = i + 1; //@ nowarn Modifies;
		return i;
	}

	//@ modifies i;
	//@ modifies ar[10];
	//@ modifies j,a.ii,a.jj,ar,ar[1];
	//@ ensures \result == \old(i) + 1;
	//@ ensures ModA.ii == \old(ModA.ii);
	//@ ensures a.jj == \old(a.jj);
	//@ ensures b.jj == \old(b.jj);
	public int mmm() {
		i = i + 1;
		ar[10]=20;
		return i;
	}

	public Mod();
}

class ModA {
	public ModA();

	public static int ii;
	public int jj;
}
