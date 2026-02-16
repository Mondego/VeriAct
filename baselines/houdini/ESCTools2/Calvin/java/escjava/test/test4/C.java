/* Copyright Hewlett-Packard, 2002 */


public class C {
    public static void test1() {
	int x;
	if( false ) {
	    x = 1;
	} else {
	    x = 2;
	}

	//@ assert x==2;
    }

    public static void test2() {
	int x=0;
	x=x+1;
	if( true ) x=x+2;
	if( false ) x=x+4;
	//@ assert x==3;
    }

  public static void test03() {
      int[] x = null;
      //@ assert x instanceof Object;
  }

  public static void test04() {
      int[] x = new int[3];
      //@ assert x instanceof Object;
  }
}
