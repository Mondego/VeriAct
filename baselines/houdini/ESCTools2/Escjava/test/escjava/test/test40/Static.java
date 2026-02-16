class Static {

    static /*@ non_null @*/ String s = "hello";


    void doit() {
	//@ assert s != null

	s = null;			// error
    }
}
