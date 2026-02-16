class M2{

    /*
     * ensures \result >= 0;
     */
    static public int f(/*@ non_null @*/ Integer x,/*@ non_null @*/ Integer y){

	int a = x.intValue();
	int b = y.intValue();

	return a + b;

    }

}
