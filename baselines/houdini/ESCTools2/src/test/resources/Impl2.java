public class Impl2 implements IntList {

	private int[] list = new int[100];
	private int size; //@ in length;

	//@ represents length = size;
	//@ represents isEmpty = (size == 0);

	public void shrink() {
		size = 1;
	} // FAILS

	public void shrinkA() {
		size = 1;
	} // OK

	public void clear() {
		size = 0;
	}

	//@ ensures isEmpty && length == 0;
	public Impl2() {
		size = 0;
	}
}

interface IntList {

	//@ public instance model boolean isEmpty;
	//@ public instance model int length;

	//@ modifies length;
	//@ ensures \not_modified(isEmpty);
	public void shrink();

        //@ requires !isEmpty;
	//@ modifies length;
	//@ ensures \not_modified(isEmpty);
	public void shrinkA();

	//@ modifies length,isEmpty;
	//@ ensures isEmpty;
	//@ ensures length == 0;
	public void clear();
}
