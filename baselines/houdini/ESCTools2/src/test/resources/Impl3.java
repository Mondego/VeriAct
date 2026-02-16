public class Impl3 implements IntList {

	private int size; //@ in length;

	//@ represents length = size;

	public void shrink() {
		size = 1;
	}

	public void clear() {
		size = 0;
	}  // ERROR - no represents clause for isEmpty
}

interface IntList {

	//@ public instance model boolean isEmpty;
	//@ public instance model int length;

	//@ modifies length;
	//@ ensures \not_modified(isEmpty);
	public void shrink();

	//@ modifies length,isEmpty;
	//@ ensures isEmpty;
	//@ ensures length == 0;
	public void clear();
}
