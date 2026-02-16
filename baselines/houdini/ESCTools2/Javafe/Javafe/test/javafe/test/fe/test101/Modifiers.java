/**
 ** These files checks which modifiers are legal for package member
 ** and top level classes and interfaces.
 **/

/*
 * Test package-member classes:
 */
public class      
    PublicClass {}            // Error because doesn't match filename
private class                 // Error: not private <p-m class>
    PrivateClass {}           
protected class               // Parse error: protected <p-m class>
    ProtectedClass {}                
static class                  // Parse error: static <p-m class>
    StaticClass {}
final class
    FinalClass {}
synchronized class            // Parse error: synchronized <p-m class>
    SyncClass {}
volatile class                // Parse error: volatile <p-m class>
    VolaClass {}
transient class               // Parse error: transient <p-m class>
    TranClass {}
native class                  // Parse error: native <p-m class>
    NativeClass {}
abstract class
    AbstClass {}

// Summary: the only permitted modifiers are: public final abstract


/*
 * Test test package-member interfaces:
 */
public interface      
    PublicInterface {}            // Error because doesn't match filename
private interface                 // Error: not private <p-m interface>
    PrivateInterface {}
protected interface               // Parse error: protected <p-m interface>
    ProtectedInterface {}                
static interface                  // Parse error: static <p-m interface>
    StaticInterface {}
final interface                   // Error: final <p-m interface>
    FinalInterface {}             
synchronized interface            // Parse error: synchronized <p-m interface>
    SyncInterface {}
volatile interface                // Parse error: volatile <p-m interface>
    VolaInterface {}
transient interface               // Parse error: transient <p-m interface>
    TranInterface {}
native interface                  // Parse error: native <p-m interface>
    NativeInterface {}
abstract interface
    AbstInterface {}

// Summary: the only permitted modifiers are: public abstract


/*
 * Test test top-level, non-package-member, classes:
 */
class Top {
    public class      
	PublicClass {}
    private class
	PrivateClass {}
    protected class      
	ProtectedClass {}                
    static class
	StaticClass {}
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

// Summary: the only permitted modifiers are: public private protected
// static final abstract


/*
 * Test test top-level, non-package-member, interfaces:
 */
class Top2 {
    public interface      
	PublicInterface {}
    private interface
	PrivateInterface {}
    protected interface      
	ProtectedInterface {}                
    static interface
	StaticInterface {}
    final interface                    // Error: interfaces can't be final
	FinalInterface {}             
    synchronized interface             // Error: inner interfacees can't be ...
	SyncInterface {}     // volatile, transient, native, or synchronized   
    volatile interface                // Error: inner interfacees can't be ...
	VolaInterface {}              
    transient interface               // Error: inner interfacees can't be ...
	TranInterface {}              
    native interface                  // Error: inner interfacees can't be ...
	NativeInterface {}            
    abstract interface
	AbstInterface {}
}

// Summary: the only permitted modifiers are: public private protected
// static abstract

