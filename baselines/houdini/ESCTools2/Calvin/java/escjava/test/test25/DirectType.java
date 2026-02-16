/* Copyright Hewlett-Packard, 2002 */

class DirectType extends Throwable {}

class DirectTypeSub extends DirectType {}

class DirectTypeSub2 extends DirectTypeSub {}


class DirectTypeUser {

    // make sure we pull in DirectTypeSub2 <: DirectType

    // (tests supertype closing, direct type ref...)
    void foo(DirectTypeSub2 x) {
	(DirectType)x;			// no error
    }


    //@ invariant ptr!=null;
    DirectTypeSub2 ptr = new DirectTypeSub2();

    // similar, but instead get type from field range:
    void bar() {
	(DirectType)ptr;		// no error
    }


    // similar, but instead get type from constructor return type:
    void baz() {
	(DirectType)(new DirectTypeSub2());		// no error
    }


    // similar, but instead get type from new array expression:
    void baz2() {
	(DirectType[])(new DirectTypeSub2[0]);		// no error
    }


    // similar, but instead get type from method return type:
    DirectTypeSub2 getit() { return new DirectTypeSub2(); }

    void quux() {
	(DirectType)(getit());		// no error
    }


    // similar, but instead get type from method throws type:
    void throwit() throws DirectTypeSub2 {}

    void beep() {
	try {
	     throwit();
	} catch (DirectType x) {}
    }					// no (unexpected exception) error
}
