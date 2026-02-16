public class Client5 {

	//@ requires list != null;
	//@ modifies list.length;
	//@ ensures \not_modified(list.isEmpty);  // CURRENT ERROR 
	public void unique(IntList list) {
		list.unique();
	}

	//@ requires list != null;
	//@ modifies list.length,list.isEmpty;
	//@ ensures \not_modified(list.isEmpty);  // ERROR
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
	public void clear();
}
