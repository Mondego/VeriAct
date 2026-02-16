public class ModN {

	public int i,j,k,m;

	//@ modifies i;
	//@ modifies j;
	//@ also
	//@ modifies i,k;
	void mok() {
		i = 0;
	}

	//@ modifies i;
	//@ modifies j;
	//@ also
	//@ modifies i,k;
	void m() {
		j = 0;
	}

	//@ modifies i;
	//@ modifies j;
	//@ also
	//@ modifies i,k;
	void mm() {
		k = 0;
	}

	//@ modifies i;
	//@ modifies j;
	//@ also
	//@ modifies i,k;
	void mmm() {
		m = 0;
	}

	//@ requires i>=0;
	//@ modifies j;
	//@ also
	//@ requires i<=0;
	//@ modifies k;
	void p() {
		if (i > 0) j = 0;
		if (i < 0) k = 0;
		if (i == 0) j = 0;
	}

	//@ requires i>=0;
	//@ modifies j;
	//@ also
	//@ requires i<=0;
	//@ modifies k;
	void pp() {
		if (i==0) k=0;
	}

	int[] a;
	//@ modifies a;
	//@ also
	//@ modifies i;
	void pa() {
		a = null;
	}
}
