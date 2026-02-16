class D{

    /*@
      @ ensures \result == true;
      @*/
    static public boolean f(int x){

	Integer i1 = new Integer(x);
	Integer i2 = i1;

	return i1 == i2;

    }

}
