class Neg {

    public static void main(String[] args) {
	out(this.class);			// parse error

    }

    static void out(Class c) {
	System.out.println(c);
    }
}
