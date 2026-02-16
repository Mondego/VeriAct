//#FLAGS: -nowarn Modifies

public class Mod2 {


	public static int a,b;
	public static int i,j,k;
	public int ii,jj,kk;

	//@ ensures j == 3+a;    // WARNING
	public void m() {
		j = 0;		//@ nowarn Modifies;
		k = 0;		//@ nowarn Modifies;
		mm(10);
		a = a + 3;		//@ nowarn Modifies;
		b = a+a;		//@ nowarn Modifies;
	}

	//@ ensures jj == 3+a;   // WARNING
	public void m1() {
		j = 0;		//@ nowarn Modifies;
		k = 0;		//@ nowarn Modifies;
		mm(10);
		a = a + 3;		//@ nowarn Modifies;
		b = a+a;		//@ nowarn Modifies;
	}

	//@ ensures k == 4+\old(a);  // WARNING 
	public void m2() {
		j = 0;		//@ nowarn Modifies;
		k = 0;		//@ nowarn Modifies;
		mm(10);
		a = a + 3;		//@ nowarn Modifies;
		b = a+a;		//@ nowarn Modifies;
	}

	//@ modifies i,j,ii,jj;
	//@ ensures j == 3+a+0*\old(j); // WARNING
	//@ ensures jj == 3+\old(a); // OK
	public void mm(int q) {
		i = 2+a;
		ii = 2+q;
		j = 3+a;
		jj = 3+a;
		a = a + 1;		//@ nowarn Modifies;
		kk = 4+a;		//@ nowarn Modifies;
		k = 4+q;		//@ nowarn Modifies;
	}

	public Mod2();
}
