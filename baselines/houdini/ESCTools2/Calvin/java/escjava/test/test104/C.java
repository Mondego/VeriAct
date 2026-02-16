/* Copyright Hewlett-Packard, 2002 */

class C {

    //@ global_invariant (mx == null && my == null) ==> x == y

    //@ ghost /*@ guarded_by mx == \tid */ public Thread mx

    //@ ghost /*@ guarded_by my == \tid */ public Thread my

    //@ guarded_by mx == \tid
    int x;

    //@ guarded_by my == \tid
    int y;

    void goo() {
	//@ assume mx == null
    }

    void loo() {
	{{
	    //@ assume mx == null
	    //@ set mx = \tid
	}}
	    
	x = 0;
	    
	//@ assert x == 0

	//@ assert mx == \tid
	
	//@ set mx = null
    }

    /*@ modifies x, y */
    /*@ performs (x, y) { x == \old(x) + 1 && y == \old(y) + 1 } */
    void boo() {
	{{
	    //@ assume mx == null
	    //@ set mx = \tid
	}}

	{{
	    //@ assume my == null
	    //@ set my = \tid
	}}

	x++;

	y++;
	
	//@ set mx = null

	//@ set my = null
    }

}
