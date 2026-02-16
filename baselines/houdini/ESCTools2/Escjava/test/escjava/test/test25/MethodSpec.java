class MethodSpec {

    //@ invariant x<=0;
    int x;

    //@ requires arg<=x;
    int foo(int arg) { return 1; }

    //@ requires arg<=x;
    int foo2(int arg) { return 1; }
}


class MethodSpecSub extends MethodSpec {

    /*
     * Ensure we see references in method declarations derived specs
     */

    //@ also ensures \result<=0;
    int foo(int arg) { return arg; }

    //@ also
    //@ requires arg <= x;
    //@ ensures \result<=0;
    int foo2(int arg) { return arg; }
}
/*
Note - ESC/Java combined the specs for foo as
	requires arg<=x;
	ensures \result <=0;
which could be verified.
ESC/Java2 combines them as
	requires arg<=x;
    also
	ensures \result<=0;
which is
	requires true;
	ensures \result <= 0;
and is not verifiable
*/
