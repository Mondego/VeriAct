/* Copyright Hewlett-Packard, 2002 */

class MethodCall {

    //@ invariant x>=0;
    int x;

    //@ ensures \result==x;
    int getx() { return x; }

    //@ requires arg<x;
    void lessx(int arg) {}
}

class MethodCallSub extends MethodCall {

    int y;

    //@ ensures y==x+1;
    MethodCallSub() {
	y = x+1;
    }

    int getx() { return super.getx(); }

    void lessx(int arg) {}
}


class MethodCallSubUser {

    /*
     * Make sure we see references in (derived) specs of methods called
     */

    // Test ensures:
    //@ requires MCS!=null;
    void foo(MethodCallSub MCS) {
	int z = MCS.getx();
	//@ assert z>=0;			// no error
    }

    // Test requires:
    //@ requires MCS!=null;
    void bar(MethodCallSub MCS) {
	MCS.lessx(-1);			// no error
    }
    //@ requires MCS!=null;
    void bar2(MethodCallSub MCS) {
	MCS.lessx(1);			// error!
    }


    /*
     * Make sure we see references in specs of constructors called via new
     */
    void baz() {
	MethodCallSub mcs = new MethodCallSub();

	//@ assert mcs.y>=1;		// no error
    }
}

class MethodCallSubUser1 extends MethodCallSub {

    /*
     * Make sure we see references in specs of constructors called directly
     */
    //@ ensures y>=1;			// no error
    MethodCallSubUser1() {
	this();
    }
}

