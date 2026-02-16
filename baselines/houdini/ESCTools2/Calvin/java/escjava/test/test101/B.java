/* Copyright Hewlett-Packard, 2002 */

class B {
    // test pred abs

    static boolean t;
    static boolean t() {
	return true;
    }

    static Object a,b;

    // test loop context
    static void b1() {
	//@ assume a != null;
	//@ loop_predicate a != null;
	while( t() ) {
	    a = a;
	    a.toString();
	}
    }

    // test preconditions
    //@ requires a != null;
    static void b2() {
	//@ loop_predicate a != null;
	while( t() ) {
	    a = a;
	    a.toString();
	}
    }

    // test loop guard
    static void b3() {
	//@ assume a != null;
	//@ loop_predicate a != null;
	while( t() && b != null ) {
	    a = b; b=b;
	}
	a.toString();
    }

    // test invariants
    static void b4() {
	//@ assume a != null;
	//@ assume b != null;
	//@ loop_predicate a != null;
	//@ loop_invariant b != null;
	while( t() ) {
	    a = b; b=b;
	}
	a.toString();
    }

    // test nested loops
    static void b5() {
	//@ assume a != null;
	//@ loop_predicate a != null;
	while( t ) {
	    a = a;
	    //a.toString();
	    //@ loop_predicate a != null;
	    while( t ) {
		a = a;
		//a.toString();
	    }
	}
	a.toString();
    }



}

class E {

    static int i;
    //@ invariant i>=0;

    static void foo() {

	//@ loop_predicate i >= 0

	for(int j=0; j<10; j++) {
	    i++;
	}
	//@ assert i >= 0;

    }
}
