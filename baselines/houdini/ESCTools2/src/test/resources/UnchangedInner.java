/**
 ** Test escjava's reasoning about member inner classes
 **/


/*
 * Test modifies clauses:
 */
public class UnchangedInner {
    int x;

    class Inner {
	Inner() {}

	//@ modifies x;
	Inner(int y) {}


	//@ modifies x;
	void modify() { x = 10; }
    }

    void test() {
	x = 3;
	Inner I = new Inner();
	//@ assert x>0;
	I.modify();
	//@ assert x>0 ;        // error
    }

    void test2() {
	x = 3;
	Inner I = new Inner(3);
	//@ assert x>0 ;        // error
    }

}
