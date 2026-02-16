/**
 ** Test anonymous class declarations placed inside superclass
 ** constructor calls.
 **/

class Top {
    Top() {}
    Top(Object x) {}
}

//
// Check references to instances via implicit C.this.f...
//
class InsideCall extends Top {
    int target;

    // zero levels deep:
    InsideCall(char c) { super(new Object() { int q; }); }
    InsideCall() { super(new Object() { int q = target; }); }  // error

    // One level deep:
    class Inner extends Top {
	Inner() {
	    super(new Object() { int q = target; });    // error
	}
    }

    // Two levels deep:
    class Middle {
	class InnerMost extends Top {
	    InnerMost() {
		super(new Object() { int q = target; });  // error
	    }
	}
    }
}
