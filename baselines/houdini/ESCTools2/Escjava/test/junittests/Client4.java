public class Client4 {

	//@ requires list != null;
	//@ modifies list.length;
	//@ ensures \not_modified(list.isEmpty);  // ERROR - insufficient info
	public void unique(IntList list) {
		list.unique();
	}

	//@ requires list != null;
	//@ modifies list.length,list.isEmpty;
	public void clear(IntList list) {
		list.clear();
	}


}

interface IntList {

	//@ public instance model boolean isEmpty;
	//@ public instance model int length;

	//@ represents isEmpty \such_that isEmpty <==> (length == 0);

	//@ modifies length;
	public void unique();

	//@ modifies length,isEmpty;
	//@ ensures length == 0;
	//@ ensures isEmpty;
	public void clear();
}
