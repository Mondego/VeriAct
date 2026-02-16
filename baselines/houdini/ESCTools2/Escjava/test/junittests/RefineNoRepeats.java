public class RefineNoRepeats {

	//@ ghost boolean b;
	//@ model float f;
	double d;

	RefineNoRepeats();
	RefineNoRepeats(int i);

	void m();
	void m(int i);
	//@ model void n();

	public class A {}
	public interface B {}
}
