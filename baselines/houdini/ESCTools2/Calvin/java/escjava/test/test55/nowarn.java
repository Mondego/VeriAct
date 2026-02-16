/* Copyright Hewlett-Packard, 2002 */


class C {
    public int i;
}

class D {
    int m(C c) {
	return c.i; //@ nowarn Null
    }
}
