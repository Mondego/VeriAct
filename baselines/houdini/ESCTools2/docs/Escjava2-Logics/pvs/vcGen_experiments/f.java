class f {

    /*@
      @ requires y!= 0;
      @*/
    static public int f(int x,int y){

	return x/y;

    }

    /*@
      @ ensures \result.intValue() >= 0;
      @*/
    static public Integer g(int x){

	if(x > 0)
	    return new Integer(x);
	else
	    return new Integer(0);
    }

}
