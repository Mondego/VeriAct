/**
 ** Test access control between nested classes
 **/

class Top {

    private long t;

    static class A {
	static private long a;

	void foo(B b) {
	    long r = b.b;
	}
    }

    class B {
	private long b = t + A.a;         // no error

	void bar(B b2) {
	    long s = this.b + t + b2.b;   // no error
	}
    }
}

class C {
    static long q = Top.A.a;              // error
}
