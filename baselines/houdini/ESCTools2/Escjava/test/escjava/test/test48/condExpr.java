
class C {
    
    C() {}

    static int m(int i)
    //@ ensures \result == 0
    {
	int result;
	result = (i == 1 ? 1 : 0);
	return result;
    }
}
