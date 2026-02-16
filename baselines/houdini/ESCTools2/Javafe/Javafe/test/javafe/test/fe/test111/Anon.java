/**
 ** Test anonymous classes, part I:
 **/


/*
 * Basic tests:
 */

class Anon {
    class Top {
	int x;

	Top() {}

	Top(char y) {}
    }

    interface ITop {
	int q = 0;
    }

    abstract class ATop {
	abstract void foo();
    }


    void m() {
	// Verify checking sub-type:
	Top t1 = new Top() {
	    int y = false;      		// error
	};

	// Verify inheriting from supertype:
	Top t2 = new Top() { int y = x+1; };

	// Check constructor matching:
	Top t3 = new Top(7) { int y = x+1; };   // error: wrong constructor

	// Make sure can implement an inteface:
	ITop t4 = new ITop() { int w = q; };
	ITop t5 = new ITop(7) {};   		// error: wrong constructor

	// Ditto an abstract class:
	ATop a1 = new ATop() { };       // error which we intentionally ignore 
	ATop a2 = new ATop() { void foo() {} };

	// Check subtype is in proper environment:
	final int z = 3;
	Top t6 = new Top() { int y = z+1; };
    }
}


/*
 * Test with inner class as a supertype:
 */

class Anon2 {
    class Inner {
	class Top {}
    }

    void m() {
	Inner I = new Inner();
	Inner.Top t = I.new Top() { int x; };
    }
}


/*
 * Test types of resulting object
 */

class Anon3 {
    class Target {}
    interface I {}

    void m() {
	Target x = new Target() { };
	I y = new I() {};
    }
}
