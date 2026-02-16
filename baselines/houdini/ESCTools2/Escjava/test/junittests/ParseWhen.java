public class ParseWhen {

	public boolean b,bb;

	//@ requires b;
	//@ when bb;
	public void m();

	//@ requires bb;

	public void n();
}
