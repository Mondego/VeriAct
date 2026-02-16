public class ParseRepresents {

	private int length;

	//@ public model int size;

	//@ private represents size <- length + 1;

	//@ public model boolean empty;
	//@ private represents empty \such_that length == 0;

	//@ ensures size > 1;
	public void m() {
		length = 1;  //@ nowarn Modifies;
	}

	//@ ensures size > 1;
	public void p() {
		length = 0; //@ nowarn Modifies;
	}
}
