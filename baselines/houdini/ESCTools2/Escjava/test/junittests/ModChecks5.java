public class ModChecks5 {

	public int i,ii;

	public int j; //@ in i;

	//@ modifies i;
	void m() {
		j = 0;
		i = 0;
	}

	public int k,kk; //@ in i;

	//@ modifies i;
	void mm() {
		k = 0;
		kk = 0;
	}

	public int kkk; //@ in i,ii;

	//@ modifies ii;
	void mmm() {
		kkk = 0;
	}

	//@ modifies i;
	void mmmm() {
		kkk = 0;
	}
}
