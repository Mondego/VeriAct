public class ModelMethod {

	//@ requires i < 0;
	//@ ensures \result > 0;
	//@ model public int m(int i);

	public void n();

	//@ model public ModelMethod(int i) {}
}
