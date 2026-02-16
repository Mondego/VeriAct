/**
 ** Test if redeclaring a local variable in a catch clause is an error.
 **/

class TestCatch {

    void foo() {
	int x;

	try {
	} catch (Exception x) {}    // Error: x redefined
    }
}

