public class Client9 {

	//@ requires list != null;
	//@ modifies list.length;
	//@ ensures \not_modified(list.isEmpty);
	public void unique(IntList list) {
		list.unique();
	}

	//@ requires list != null;
	//@ modifies list.length; // MISSING list.isEmpty;
	//@ ensures list.isEmpty;
	//@ ensures list.length == 0;
	//@ ensures \old(list.isEmpty);  // ?? OK BECAUSE OF MISSING MODIFIES CLAUSES
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

	//@ modifies length; // MISSING isEmpty;
	//@ ensures isEmpty;
	//@ ensures length == 0;
	public void clear();
}
