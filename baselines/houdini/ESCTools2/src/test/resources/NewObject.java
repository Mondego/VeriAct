public class NewObject {
	NewObject();

	static public NewObjectA o;
	static public int[] b;

	//@ ensures (new NewObjectA()).len == 7;
	public void m() {}

	//@ ensures (new NewObjectA()) != null;
	//@ ensures \typeof(new NewObjectA()) == \type(NewObjectA);
	//@ ensures (new NewObjectA()) != \result;
	//@ ensures (new NewObjectA()) != o;
	//@ ensures (new NewObjectA()) != (new NewObjectA());
	public NewObjectA mm() {
		o = new NewObjectA(); //@ nowarn Modifies;
		return new NewObjectA();
	}

	//@ ensures (new int[5]) != null;
	//@ ensures (new int[5]).length == 5;
	//@ ensures \typeof(new int[5]) == \type(int[]);
	//@ ensures (new int[6]) != \result;
	//@ ensures (new int[6]) != b;
	public int[] mmm() {
		int[] a = new int[6];
		return a;
	}

	public void m4() {
		int[] a = new int[5];
		Object[][] aa = new Object[5][6];
		//@ assert a.length == 5;
		//@ assert a != null;
		//@ assert \typeof(a) == \type(int[]);
	}

}

class NewObjectA {
	//@ pure
	NewObjectA();

	//@ invariant len == 7;

	public int len;
}
