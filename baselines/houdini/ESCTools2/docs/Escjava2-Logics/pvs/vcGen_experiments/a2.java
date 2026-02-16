class A2{

    /*@ requires x >= 0 && y >= 0;
      @  ensures \result >= 0;
      @*/
    static public int f(int x,int y){
	
	return x + y;
    }

	

}
