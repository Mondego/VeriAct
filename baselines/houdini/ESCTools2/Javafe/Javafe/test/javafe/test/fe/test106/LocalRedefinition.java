/**
 ** Test that we handle local redefinition errors properly
 **/

/*
 * First test redeclaring formal parameters:
 *
 * (local variables have the same behavior.)
 */

class B {
    void f (int i) { // declare parameter
	class C {
	    void f (int i) { // error - hides parameter // NOT AN ERROR in javac
	    }
	}
	
	class D {
	    int i; // OK; *field* hides parameter
	    void f (int i) { // OK - hides *field*
	    }
	}
    }
}


/*
 * Then test redeclaring local types:
 *
 * A local type definition can hide a type member, but not another
 * local type defintion (even in another class).
 */

class Top {
    void m() {
	class T {};
	class T {};         // error
    }

    class R{};
    void r() {
	class R {};         // no error
    }	

    void s() {
	class S{};
	class T{
	    void t() {
		class S{};   // error...
	    }
	}
    }

    void z() {
	class S{};
	class T{
	    class S{};
	    void t() {
		class S{};     // no error
	    }
	}
    }
}
