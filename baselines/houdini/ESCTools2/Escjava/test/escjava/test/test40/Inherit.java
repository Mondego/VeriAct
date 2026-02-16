class A {
    void put(/*@ non_null @*/ String x) { }
}

class B extends A { }

class C extends B {

    void put(String y) {
	//@ assert y != null

	y = null;				// error
    }
}
