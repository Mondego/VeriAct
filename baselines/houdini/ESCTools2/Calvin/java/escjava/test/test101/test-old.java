/* Copyright Hewlett-Packard, 2002 */

class c {
    static int x,y;
    
    //@ ensures x+y > \old(x+y)
    //@ modifies y
    void m() {
	x++;
    }

    void mi() {
	y=0;
	//@ loop_invariant x+y == \old(x);
	for(;;) {
	    x++;
	    y--;
	}
    }
    void mp() {
	y=0;
	int oldx = x;
	//@ loop_predicate x+y == \old(x);
	for(;;) {
	    x++;
	    y--;
	    //@ assert x+y==oldx;
	}
    }
}
