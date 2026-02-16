class D {

    static void n(Object[] a, Object[] b)
	//@ requires a.length > 0;
	//@ requires a != null;
	//@ requires b.length > 0;
	//@ requires b != null;
	//@ requires \typeof(b) <: \typeof(a);
	{
	    // assert \type(Object) <: \elemtype(\typeof(a))
	    //@ assert a[0] == null | \typeof(a[0]) <: \type(Object);
	    //@ assert a[0] instanceof Object;
	    a[0] = b[0];
	}
}
