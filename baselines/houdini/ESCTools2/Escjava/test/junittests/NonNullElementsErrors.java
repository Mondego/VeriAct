// Tests parsing/typechekcing errors with \nonnullelements

public class NonNullElementsErrors {

	public int i;
	public Object[] a;

	//@ ensures \nonnullelements(i);  // ERROR
	//@ ensures \nonnullelements(a);
	//@ ensures \nonnullelements(new int[10]); // ERROR
	//@ ensures \nonnullelements(new Object[2]);
	//@ ensures \nonnullelements(2*i); // ERROR
	//@ ensures \nonnullelements(i>0 ? a:a);
	void m();
}

