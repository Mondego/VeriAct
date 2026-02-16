/* Copyright Hewlett-Packard, 2002 */

class C {

    C() {}
    
    static int m()
	//@ ensures \result == 0;
	{
	    return n();
	}

    static int n()
	//@ ensures \result == 0;
	{
	    return 0;
	}

}
