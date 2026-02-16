/* Copyright Hewlett-Packard, 2002 */

class C {
    
    /*@ non_null*/ Integer i;
    /*@ non_null*/ Integer j;
    

    C() {
	int k = init(4);
    }

    //@ ensures \result == 4
    /*@ helper */ private int init(int k) {
	i = new Integer(13);
	j = new Integer (55);
	return k;
    }

    void m(int x) {
        int k = init(x);
    }  // postcondition violation
}
