
class C {
    
    C() throws Exception 
    {
	throw new Exception();
    }

    int m(int i)
    //@ ensures \result == 0
    {
	try {
	    C c = new C();
	    return 0;
	}
	catch (Exception e) {
	    return 1;
	}
    }
}
