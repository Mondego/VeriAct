class E{

    /*@
      @ ensures \result == true;
      @*/

    boolean f(){

        Integer[] t = new Integer[2];

	Integer x = new Integer(1);

	t[0] = new Integer(1);
	t[1] = t[0];

	return t[0] == t[1];

    }

}
