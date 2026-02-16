class P {

    static public int f(Integer x) {

	//@ assume x != null;

	int y = x.intValue();

	return y;

    }

}
