/* Copyright Hewlett-Packard, 2002 */

class Trans {

    //@ invariant x>=0;
    int x;
}

class Trans2 {

    //@ invariant ptr!=null;
    Trans ptr = new Trans();

    //@ invariant y == ptr.x;
    int y = ptr.x;
}


class Trans2User {

/*
 * Ensure invariants are pulled in transitively when needed:
 */

    //@ requires X!=null;
    //@ ensures \result>=0;			// no error
    int foo(Trans2 X) {
	return X.y;
    }
}
