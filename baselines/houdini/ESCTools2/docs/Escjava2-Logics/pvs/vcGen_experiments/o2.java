class O2 {

    /*@
      @ ensures \result >= \old(x);
      @*/
    static public int f(int x, Integer y){

	int b = y.intValue();

	return x + b;

    }

}
