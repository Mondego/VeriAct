class C {

    void foo() {	
	for(int i = 0; i < 10; i++) {
	    //@ assert i >= 0;
	}
    }

    void goo1() {
	int k = 1;

	for(int i = 0; i < 10; i++) {
	    k++;
	}	
	//@ assert k >= 1;
    }

    void goo2() {
	int k = 1;

	for(int i = 0; i < 10; i++) {
	    k--;
	}	
	//@ assert k <= 1;
    }

    Object [] ta = new Object[10];
    //@ invariant ta != null
    
    void loo() {
	Object[] a = ta;
	
	for (int i = 0; i < a.length; i++)
	    a[i] = null;
	//@ assert (\forall int j; 0 <= j && j < a.length ==> a[j] == null);
    }
}
