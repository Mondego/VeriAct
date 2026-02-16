/**
 ** Test a corner case of Java 1.1: local class declarations contained
 ** in static members.
 **/

class LocalClassInStaticMember {

    int i = 0;
    static int s = 1;

    static void m() {
	final int l = 3;

	class Local {      // Enclosing instances: none
	    int f = i;       // error: outside instance fields not accessible
	    int g = s;       //        static fields are ok, though
	    int x = l;       //        ditto outside final local variables

	    static int w=1;  // error: local types can't contain static members

	    class Inner {  // Enclosing instances: Local
		int q = f;   // we have a Local enclosing instance so ok...
		int r = i;   // error: still can't access i field
	    }
	}

	Local e = new Local();  // ok since don't need an enclosing instance

	class Local2 extends Local {
	    Local2() {}         // ok for same reason
	}
    }
}

//* Basically same test, but using anonymous instead of block-level classes:

class LocalClassInStaticMember2 {

    int i = 0;
    static int s = 1;

    static void m() {
	final int l = 3;

	Object o = new Object() {  // Enclosing instances: none
	    int f = i;       // error: outside instance fields not accessible
	    int g = s;       //        static fields are ok, though
	    int x = l;       //        ditto outside final local variables

	    static int w=1;  // error: local types can't contain static members

	    class Inner {  // Enclosing instances: <anonymous>
		int q = f;   // we have a <anony.> enclosing instance so ok...
		int r = i;   // error: still can't access i field
	    }
	};
    }
}

