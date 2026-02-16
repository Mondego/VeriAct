class O {

    /*@
      @ ensures \result >= \old(x);
      @*/
    static public int f(int x, /*@ non_null @*/ Integer y){

	int b = y.intValue();

	if(b <= 0)
	    return x+=1;
	else
	    return x + b;

    }

}
