/**
 ** Check that (implicit) references to instance and static
 ** fields/methods are handled properly, both in disambiguation and in
 ** type checking.
 **
 ** (Here, we are not worried about *which* member
 ** disambiguation/lookup resolves to, but that we insert the C. or
 ** C.this. properly and that we check access to instance variables
 ** correctly.)
 **/

class Fields {
    static int s;
    int i;


    // First check uses from same class:
    int instance_method() { return s + i + this.i; }
    int instance_method2() {
	return Fields.s + Fields.i;   // JLS error: Fields.i (javac goofs)
    }

    static int static_method() {
	s = s + Fields.s;
	return i + this.i + Fields.i;        // error: i, this.i, & Fields.i
    }


    // Then check uses from a nested inner class:
    class InnerTop { InnerTop(int x) {} }
    class Inner extends InnerTop {
	int instance_method() { 
	    s = Fields.this.i;
	    return s + Fields.s + i;
	}
	static int static_method() {}    // error!
	// From inside a superclass constructor is also effectively static:
	Inner() { super(i); }
	Inner(int q) { super(s); }

	// And a yet more deeply nested inner class:
	class Inner2 extends InnerTop {
	    int instance_method() { 
		s = Fields.this.i;
		return s + Fields.s + i;
	    }
	    Inner2() { super(i); }
	    Inner2(int q) { super(s); }
	}
    }


    // Then check uses from a nested static class:
    static class Static {

	int instance_method() { 
	    s = Fields.this.i;    // error
	    return s + Fields.s + i;            // error: i
	}

	static int static_method() {
	    s = s + Fields.s;
	    s = Fields.this.i;    // error
	    return i + this.i + Fields.i;      // error: i, i.this, & Fields.i
	}
    }
}



class Methods {
    static int s() { return 0; }
    int i() { return 1; }


    // First check uses from same class:
    int instance_method() { return s() + i() + this.i(); }
    int instance_method2() {
	return Methods.s() + Methods.i(); // JLS error: Methods.i (javac goofs)
    }

    static int static_method() {
	int x = s() + Methods.s();
	return i() + this.i() + Methods.i();  // error: i, this.i, & Methods.i
    }


    // Then check uses from a nested inner class:
    class Inner {
	int instance_method() { 
	    Methods.this.i();
	    return s() + Methods.s() + i();
	}
	static int static_method() {}    // error!


	// And a yet more deeply nested inner class:
	class Inner2 {
	    int instance_method() { 
		Methods.this.i();
		return s() + Methods.s() + i();
	    }
	}
    }


    // Then check uses from a nested static class:
    static class Static {

	int instance_method() { 
	    Methods.this.i();    // error
	    return s() + Methods.s() + i();            // error: i
	}

	static int static_method() {
	    int x = s() + Methods.s();
	    Methods.this.i();    // error
	    return i()+this.i()+Methods.i();// error: i, i.this, & Methods.i
	}
    }
}
