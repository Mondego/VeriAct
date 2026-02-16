/* Copyright Hewlett-Packard, 2002 */


class C {
    C() {
    }

    //@ ensures \result > 0
    int m(int j) {
	return j;
    }
}
