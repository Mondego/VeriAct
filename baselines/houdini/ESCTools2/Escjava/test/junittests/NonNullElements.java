// Tests reasoning with \nonnullelements

public class NonNullElements {

	public Object[] a;

	//@ requires a.length > 5;
	//@ requires \nonnullelements(a);
	//@ ensures a[5] != null;
	public void m() {}
}
