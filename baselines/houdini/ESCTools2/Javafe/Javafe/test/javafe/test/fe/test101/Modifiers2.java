/**
 ** This file checks which modifiers are legal for member inner types.
 **/

/*
 * test inner classes:
 */
class Top {
    /*NOT static*/ class Inner {
	public class      
	    PublicClass {}
	private class
	    PrivateClass {}
	protected class      
	    ProtectedClass {}                
	static class                  // Error: Static members can only
	    StaticClass {}            // occur in interfaces and top-level
                                      // classes
	final class
	    FinalClass {}
	synchronized class            // Error: inner classes can't be ...
	    SyncClass {}              
	volatile class                // Error: inner classes can't be ...
	    VolaClass {}              
	transient class               // Error: inner classes can't be ...
	    TranClass {}              
	native class                  // Error: inner classes can't be ...
	    NativeClass {}            
	abstract class
	    AbstClass {}
    }
}

// Summary: the only permitted modifiers are: public private protected
// final abstract


/*
 * test inner interfaces:
 */
class Top2 {
    /*NOT static*/ class Inner {
	interface                    // Error: interfaces can't be members here
	    PublicInterface {}       
    }
}

// Summary: the only permitted modifiers are: public private protected
// abstract


/*
 * test inner classes inside of an interface:
 */
class Top3 {
    /*implicitly static*/ interface Inner {
	public class      
	    PublicClass {}
	private class              // interface "fields" can't be private or
	    PrivateClass {}        // protected
                                   
	protected class            // interface "fields" can't be private or
	    ProtectedClass {}      // protected
                                   
	static class
	    StaticClass {}         // OK!!
	final class
	    FinalClass {}
	synchronized class         // Error: inner classes can't be ...
	    SyncClass {}              
	volatile class             // Error: inner classes can't be ...
	    VolaClass {}            
	transient class            // Error: inner classes can't be ...
	    TranClass {}              
	native class               // Error: inner classes can't be ...
	    NativeClass {}            
	abstract class
	    AbstClass {}
    }
}

// Summary: the only permitted modifiers are: public static 
// final abstract


/*
 * test inner interfaces inside of an interface:
 */
class Top4 {
    /*implicitly static*/ interface Inner {
	public interface      
	    PublicInterface {}
	private interface          // interface "fields" can't be private or
	    PrivateInterface {}    // protected
                                   
	protected interface        // interface "fields" can't be private or
	    ProtectedInterface {}  // protected
                                   
	static interface
	    StaticInterface {}     // OK!!
	final interface            // interfaces can't be final
	    FinalInterface {}      
	synchronized interface     // Error: inner interfacees can't be ...
	    SyncInterface {}       
	volatile interface         // Error: inner interfacees can't be ...
	    VolaInterface {}       
	transient interface        // Error: inner interfacees can't be ...
	    TranInterface {}       
	native interface           // Error: inner interfacees can't be ...
	    NativeInterface {}     
	abstract interface
	    AbstInterface {}
    }
}

// Summary: the only permitted modifiers are: public static 
// abstract
