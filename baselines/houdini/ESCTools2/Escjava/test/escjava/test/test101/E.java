class E {

    static int i;
    //@ invariant i >= 0;

    static void foo() {

	//@ loop_predicate i >= 0;

	for(int j = 0; j < 10; j++) {
	    i++;
	}
	//@ assert i >= 0;

    }
}
