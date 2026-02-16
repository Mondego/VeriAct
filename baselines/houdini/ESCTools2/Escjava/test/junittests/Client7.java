public class Client7 {

	//@ requires list != null;
	//@ modifies list.length;
	//@ ensures \not_modified(list.isEmpty);
	public void unique(IntList list) {
		list.unique();
	}

	//@ requires list != null;
	//@ modifies list.length,list.isEmpty;
	//@ ensures list.isEmpty;
	//@ ensures list.length == 0;
	public void clear(IntList list) {
		list.clear();
	}


}
interface IntList {

	//@ public instance model boolean isEmpty;
	//@ public instance model int length;


	//@ modifies length;
	//  Implicit \not_modified(isEmpty);
	public void unique();

	//@ modifies length,isEmpty;
	//@ ensures isEmpty;
	//@ ensures length == 0;
	public void clear();
}
