/* Copyright Hewlett-Packard, 2002 */

class E {
    //@ ghost /*@ elems_guarded_by[i] mx[i] == \tid */ public int -> Thread mx

    //@ ghost /*@ elems_guarded_by[j] mx[j] == \tid */ public int -> int x

    //@ ghost /*@ guarded_by my == \tid */ public Thread my

    //@ guarded_by my == \tid 
    int y;

    //@ global_invariant (\forall int i; mx[i] == null && my == null ==> x[i] > y)

    void foo(int n) {
	{{ 
	    //@ assume mx[n] == null
	    //@ set mx[n] = \tid
	}}

	//@ set x[n] = x[n] + 1

	//@ set mx[n] = null
    }

    void goo(int n) {
	{{ 
	    //@ assume mx[n] == null
	    //@ set mx[n] = \tid
	}}

	{{
	    //@ assume my == null
	    //@ set my = \tid 
	}}

	//@ set x[n] = x[n] + 1
	
	y = y - 1;

	//@ set my = null

	//@ set mx[n] = null
    }
}
