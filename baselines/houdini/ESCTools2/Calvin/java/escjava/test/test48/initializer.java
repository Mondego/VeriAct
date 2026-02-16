/* Copyright Hewlett-Packard, 2002 */


class C {

    C() 
    //@ ensures b == false
    {
	return;
    }

    public boolean b = true || false;
}
