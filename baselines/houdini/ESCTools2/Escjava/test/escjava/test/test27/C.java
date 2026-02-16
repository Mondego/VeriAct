class C {

    static void n(Object[] a)
	//@ requires a.length > 0;
	//@ requires a != null;
	{
	    // assert \type(Object) <: \elemtype(\typeof(a))
	    //@ assert a[0] == null | \typeof(a[0]) <: \type(Object);
	    //@ assert a[0] instanceof Object;
	}

    static void m() {
	Object[] a = new Object[-1];
    }
}
