// Tests that assignment statements are checked against modifies clauses.
//#FLAGS: -quiet -classpath .

public class ModChecks extends S {
	ModChecks();

	int i,k;
	static int j,jj;
	/*@ non_null */ int a[]; //@ invariant a.length > 10;
	/*@ non_null */ int b[]; //@ invariant b.length > 10;

	//@ non_null
	static ModChecks o = new ModChecks();

	//@ ghost int gi;

	//@ modifies i,j,a,a[0],gi,o.j;
	void m() {
		int zz = 9;
		i = 0;
		j = 1;
		a[0] = 7;
		a = new int[15];
		//@ set gi = 0;
		o.j = 2;
		zz = 9;
	}

	//@ requires o == this;
	//@ modifies o.i,k;
	void mm() {
		o.i = 9;
		k = 9;
		this.i = 9;
		o.k = 9;
	}
		
	//@ requires o != this;
	//@ modifies o.i,k;
	void mme() {
		o.i = 9;
		k = 9;
		this.i = 9;  // WARNING
	}
	//@ requires o != this;
	//@ modifies o.i,k;
	void mme1() {
		o.k = 9;     // WARNING
	}
		
	
	//@ modifies \everything;
	void me() {
		i = 0;		
		j = 1;	
		a = new int[15];	
		a[0] = 7;	
		//@ set gi = 0;
		o.j = 2;
	}

	//@ requires i == 0;
	//@ modifies \everything;
	void mee() {
		if (i == 0) j = 1;
		else k = 9;
	}

	//@ requires i == 0;
	//@ modifies j;
	//@ also requires i != 0;
	//@ modifies k;
	void meei() {
		if (i == 0) j = 1;
		else k = 9;
	}

	//@ requires i == 0;
	//@ modifies j;
	//@ also requires i != 0;
	//@ modifies k;
	void meei2() {
		if (i != 0) j = 1; // WARNING
		else k = 9;        // WARNING
	}

	//@ modifies a[0];
	void ma() {
		a[0] = 9;
		a[1] = 9; // WARNING
	}

	//@ modifies a[0];
	void mb() {
		b[0] = 0; // WARNING
	}

	//@ requires a == b;
	//@ modifies a[0];
	void mbb() {
		b[0] = 0; 
	}

	//@ modifies a[*];
	void maaWild() {
		a[0] = 9;
		a[1] = 9;
		b[0] = 0; // WARNING
	}
	//@ modifies a[2 .. 4];
	void maaRange1() {
		a[1] = 9; // WARNING
	}
	//@ modifies a[2 .. 4];
	void maaRangeOK() {
		a[2] = 9;
		a[3] = 9;
		a[4] = 9;
	}


	//@ modifies a[2 .. 4];
	void maaRange3() {
		a[5] = 9; // WARNING
	}

	//@ modifies j, ModChecks.jj;
	void mstatic() {
		ModChecks.j = 0;
		jj = 0;
		ModChecks.jj = 0;
		j = 0;
		this.j = 0;
		this.jj = 0;
		o.jj = 0;
		o.j = 0;
	}
	

	//@ modifies ModChecks.*;
	void mstatic2() {
		j = 0;
		jj = 0;
	}


	//@ modifies this.*;
	void mstar() {
		i = 0;
		this.k = 0;
	}

	//@ modifies o.*;
	void mstar2() {
		o.i = 0;
		if (o == this) k = 0;
	}

	//@ modifies this.*;
	void mstar3() {
		if (o == this) o.i = 0; 
	}
		
	//@ modifies o.*;
	void mstar3b() {
		if (o == this) i = 0; 
		o.i = 0;
	}
		
	//@ modifies this.*;
	void mstar4() {
		o.i = 0;  // WARNING
	}
		

	// Check that we evaluate in the prestate
	//@ requires i == 0;
	//@ modifies i,a[i],k;
	//@ also requires i > 0 && i < 5;
	//@ modifies a[*],i;
	void mpre() {
		++i;
		a[i-1] = 0;
		++i;
		if (i == 2) k = 0;
	}

	//@ modifies super.i;
	void msuper() {
		super.i = 0;
		S s = this;
		s.i = 0;
	}

	//@ modifies \nothing;
	void mnowarn() {
		i = 0; //@ nowarn Modifies;
		j = 0; //@ nowarn ;
	}

	//@ non_null
	final int[][] aa = new int[4][5];

	//@ invariant \nonnullelements(aa);
	//@ invariant aa.length > 5;
	//@ invariant (\forall int i; 0<=i && i<aa.length; aa[i].length > 5);

	//@ modifies aa[0][*];
	void mma() {
		aa[0][0] = 0;
		aa[1][0] = 0; // WARNING - FIXME
		aa[0] = new int[9]; // WARNING - FIXME
	}

	//@ modifies aa[0][0];
	void mma2() {
		aa[0][0] = 0;
		aa[1][0] = 0; // WARNING - FIXME
		aa[0] = new int[9]; // WARNING - FIXME
	}

	//@ modifies \nothing;
	void mg() {
		//@ ghost int i = 0;
		//@ set i = 0;
	}


}

class S {
	protected int i;
}
