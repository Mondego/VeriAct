public class Impl4 implements IntList {

	private int size; //@ in length;

	//@ represents length = size;
	//@ represents isEmpty = (size == 0);

	public void unique() {
		size = 1;
	}

	public void clear() {
		size = 0;
	}
}

interface IntList {

	//@ public instance model boolean isEmpty;
	//@ public instance model int length;

	//@ requires !isEmpty;
	//@ modifies length;
	//@ ensures \not_modified(isEmpty);
	public void unique();

	//@ requires !isEmpty;
	//@ modifies length; // MISSING isEmpty; 
	//@ ensures length == 0;
	//@ ensures isEmpty;  // ERROR because of missing modifies clause
	public void clear();
}
