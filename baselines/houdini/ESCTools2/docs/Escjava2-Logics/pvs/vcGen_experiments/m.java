class M{

    /*
     * ensures \result >= 0;
     */
    static public int f(/*@ non_null @*/ Integer x,/*@ non_null @*/ Integer y){

	int a = x.intValue();
	int b = y.intValue();

	//@ assume a >= 0 && b >= 0;
	
	return a + b;

    }

}
