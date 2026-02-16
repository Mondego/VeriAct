/* Copyright Hewlett-Packard, 2002 */

class D {

    static int m()
	//@ ensures \result == 0;
	{
	    return 0;
	}
}
