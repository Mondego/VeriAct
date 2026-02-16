/* Copyright Hewlett-Packard, 2002 */

class TypeChecking {

    /*@ non_null @*/ int x;	// error: non-reference type

}


class Top {
    void get(Object x) {}
}

class Bottom extends Top {
    void get(/*@ non_null @*/ Object x) {}		// error: overrides!
}


class Returns {

    abstract /*@ non_null @*/ String name();	// error

}
