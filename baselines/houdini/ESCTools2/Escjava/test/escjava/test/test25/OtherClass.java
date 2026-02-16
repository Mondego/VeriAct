class OtherClass {

    //@ invariant x==0;
    int x;

    //@ invariant y==0;
    static int y;
}


class OtherClassUser {

    /*
     * make sure we can find an invariant in another class
     * via a direct field reference:
     */

    //@ requires O != null;
    void foo(OtherClass O) {
	O.x = 1;
    }				// error!


    //@ invariant OP != null;
    OtherClass OP = new OtherClass();

    void bar() {
	OP.x = 1;
    }				// error!


    void baz() {
	OtherClass.y = 1;
    }				// error!
}
