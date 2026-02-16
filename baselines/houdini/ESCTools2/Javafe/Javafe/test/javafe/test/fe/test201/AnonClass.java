/**
 ** Test that the result type of an anonymous classes declaration (via
 ** new) is that class, not its superclass.
 **/

class AnonClass {
    void m() {
	(new AnonClass() { void p() {} }).p();        // Ok
    }
    
    void n() {
	(new AnonClass() { void p() {} }).x();        // Error
    }
}
