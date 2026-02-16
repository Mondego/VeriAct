class C {
    
    /*@ helper */ C() {}

    /*@ helper */ private int m(int i) {
	return 34;
    }
    
    private int q() {
	return 22;
    }
}

class D extends C {

    /*@ helper */ int k;

    /*@ helper */ private int q() {
	return 24;
    }

    /*@ helper */ private int m(int i) {
	return 43;
    }
}
