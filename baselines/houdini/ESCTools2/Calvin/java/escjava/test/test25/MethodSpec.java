/* Copyright Hewlett-Packard, 2002 */

class MethodSpec {

    //@ invariant x<=0;
    int x;

    //@ requires arg<=x;
    int foo(int arg) { return 1; }
}


class MethodSpecSub extends MethodSpec {

    /*
     * Ensure we see references in method declarations derived specs
     */

    //@ also_ensures \result<=0;
    int foo(int arg) { return arg; }
}
