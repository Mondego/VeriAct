/**
 ** This file checks which modifiers are legal for non-member inner types.
 **/

/*
 * test non-member inner types:
 */
class Top {
    void foo() {
	// These barf before the class keyword in javac:
	public class           // error
	    PublicClass {};
	private class          // error
	    PrivateClass {};
	protected class        // error
	    ProtectedClass {}                
	static class           // error
	    StaticClass {}
	volatile class       // error
	    VolaClass {}
	transient class      // error
	    TranClass {}              
	native class            // error   
	    NativeClass {}
	
	abstract class
	    AbstClass {}
	final class
	    FinalClass {}

	// Continued in Modifiers3a.java...
    }
}

// summary: non-member inner classes can have only abstract and final
// summary: non-member inner interfaces can have only abstract
