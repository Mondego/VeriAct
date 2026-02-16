// Tests the \is_initialized function

public class IsInitialized {
	IsInitializedA a;
	//@ ensures \is_initialized(java.util.Vector);
	//@ ensures \is_initialized(IsInitializedA);
	void m() {
		a.i=0;  //@ nowarn Modifies;
	}


}

class IsInitializedA {
	static int i;
	//@ invariant i == 0;
}
