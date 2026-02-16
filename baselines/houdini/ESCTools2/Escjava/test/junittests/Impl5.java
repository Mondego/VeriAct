
// Should have an ERROR because in clear(), isEmpty is not modified
// but its value will change

public class Impl5 implements IntList {

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
	//@ modifies length; 
	//@ ensures length == 0;
	public void clear();
}
