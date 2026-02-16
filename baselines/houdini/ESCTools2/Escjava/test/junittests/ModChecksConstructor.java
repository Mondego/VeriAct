public class ModChecksConstructor {

	//@ pure 
	public ModChecksConstructor() {
	    i = 0; //OK - pure default is this.*
	}

	int i;


	public ModChecksConstructor(int k) {
	    i = 0; // OK - default is \everything
	}


	//@ modifies \nothing;
	public ModChecksConstructor(Object o) {
	    i = 0; // WARNING
	}
}
