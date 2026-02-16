/* Copyright Hewlett-Packard, 2002 */

/**
 ** This file tests that set does a writecheck on its target and that
 ** ghost fields may have modifiers.
 **/

// First test writable_if on ghost fields...
class WritableIf {
    //@ ghost public Object mono //@ writable_if mono==null

    WritableIf() {
	Object newthing = new Object();

	//@ set mono = null;           // ok
	//@ set mono = newthing;       // ok, 1st time
	//@ set mono = newthing;       // error, 2nd time
    }

    WritableIf(int x) {
	Object newthing = new Object();

	//@ set mono = null;           // ok
	//@ set mono = newthing;       // ok, 1st time
	//@ set mono = null;           // error, 2nd time
    }
}

// Next, test non_null on ghost fields...
class NonNull {
    //@ ghost public /*@ non_null @*/ Object foo
    //@ ghost public static /*@ non_null @*/ Object s

    void foo(/*@ non_null @*/ Object x) {
	//@ assert foo!=null
	//@ set foo = x        // ok since x!=null
	//@ set foo = null     // error
	//@ set foo = foo      // ok since foo is known to be non-null
    }

    void foo2(/*@ non_null @*/ Object x) {
	//@ assert s!=null
	//@ set s = x        // ok since x!=null
	//@ set s = null     // error
	//@ set s = s        // ok since s is known to be non-null
    }
}


// Next, test monitored_by on ghost fields...
class MonitoredBy {
    public Object mutex;

    //@ ghost public Object x //@ monitored_by mutex

    void foo(Object foo) {
	//@ assert x==x        // ok: annotations don't "read" variables...
	//@ set x = foo         // error

	synchronized (mutex) {
	    //@ set x = foo     // ok since hold mutex
	}
    }
}
