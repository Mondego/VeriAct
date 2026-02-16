class F {
    /*@ non_null */ static int a[];
    void m() {
        int i;
	//@ skolem_constant int c; 
	// skolem_constant int l; 
	// skolem_constant java.lang.Object b[]; 
        
	//@ loop_predicate i >= 0;
	//@ loop_predicate c >= 0, c < i;
	//@ loop_predicate a[c] == 0;
	for(i = 0; i < a.length; i++) {
	    a[i] = 0;
	    //@ assert (\forall int j; 0 <= j && j <= i ==> a[j] == 0 );
	}
	//@ assert (\forall int j; 0 <= j && j < a.length ==> a[j] == 0 );
    }
}
