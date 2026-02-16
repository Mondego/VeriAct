public class Impl1 implements IntList {

	private int size; //@ in length;
	//@ invariant size >= 0;

	//@ represents length = size;

	public void shrink() {
		if (size != 0) size = 1;
	}

	public void clear() {
		size = 0;
	}
}

interface IntList {

	//@ public instance model boolean isEmpty;
	//@ public instance model int length;

	//@ represents isEmpty = (length == 0);

	//@ modifies length;
	//@ ensures \not_modified(isEmpty);
	// ensures (isEmpty) == \old(isEmpty);
	public void shrink();

	//@ modifies length,isEmpty;
	//@ ensures isEmpty;
	//@ ensures length == 0;
	public void clear();
}
