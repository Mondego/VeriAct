class C {
    
    /*@ helper */ C() {}

    /*@ helper */ static int m(int i) {
	return 34;
    }
}

class D extends C {

    static int q(int i) {
	return m(i);
    }

    /*@ helper */ static int m(int i) {
	return 43;
    }
}
