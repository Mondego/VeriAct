class Basic {

    /*@ non_null @*/ String x;

    Basic() { }		// error: x not initialized to a non-null value

    void doit() {
	//@ assert x != null	// no error

	x = x;			// no error

	x = null;		// error
    }


    //@ modifies x
    void changex() {}

    void call() {
	changex();

	//@ assert x != null	// no error
    }
}
