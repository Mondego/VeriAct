public class ParseRange {

	public Object[] array;

	//@ modifies array[0..1];
	//@ modifies array[0 .. 1];
	public void m();
}
