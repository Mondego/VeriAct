class C {
    Object x,y; // x non-null

    C() {
	x = new Object();
    }

    void m() {
	x.toString();	
	y.toString(); // warning
    }

    public static void main(String argv[]) {
	(new C()).m();
    }
}
