class C {

    int i;

    void m(C c, int[] j, int[] k) {
	
	//@ assume this != c;
	//@ assume null != c;
	//@ assume i >= 0;
	//@ assume j[0] >= 0;
	//@ assume j.length > 4;
	//@ assume j != null;
	//@ assume j != k;
	//@ assume k[1] >= 0;
	//@ assume k.length > 4;
	//@ assume k != null

	for(;;) {
	    c.i --;
	    //@ assert i >= 0;
	    j[1]--;
	    //@ assert j[0] >= 0;
	    //@ assert k[1] >= 0;
	    n(c,j);
	}
    }

    //@ modifies c.i, j[2];
    void n(C c, int[] j) {
    }
}
