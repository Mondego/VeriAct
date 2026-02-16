/**
 ** Test the scoping of method names in method invocations.
 **
 **
 ** The algorithm for disambiguting a method invocation N.m(...) is as
 ** follows: (remember disambiguting determines only what class or
 ** object expression the method comes from, not *which* of a set of
 ** overloaded methods it referrs to)
 **
 **
 ** (1) If N is empty, then m refers to a method of the lexically
 **     innermost enclosing class that has a method named m (counting
 **     a type's methods as appearing in all of its subtypes).
 **
 ** (2) Otherwise, attempt to resolve N as an Expr name.  If we
 **     succeed, then m denotes a method of the object that expression
 **     denotes. 
 **
 ** (3) If that fails, then N must be a TypeName that denotes a type;
 **     resolve it accordingly.  m then denotes a method of that type.
 **
 ** (4) If (1-3) all fail, then we cannot resolve the ambiguous method
 **     invocation.
 **/


/*
 * Verify that simple names refer to the innermost method definition,
 * even when method inheritance is involved.
 *
 * Note: javac enforces an additional type checking rule we don't
 * which makes the lines with a [error] an error.
 */

class Innermost {
    static int m(String x) { return 0; }

    static class I {
	static boolean m(Exception E) { return false; }

	boolean y = m(null);   // innermost method wins
    }
}
class Inheritance {
    static int m(Exception E) { return 0; }

    static class I extends withM {
	boolean y = m(null);   // [error] innermost method wins
    }
}
class withM {
    static boolean m(String x) { return false; }
}


/*
 * Verify that the # of a method's arguments have nothing to
 * do with resolving it:
 */

class Arguments {
    int m(String x) { return x.length(); }

    class I {

	boolean m() { return false; }

	void doit() {
	    boolean x = m("hello");  // Error: wrong # of arguments
	}
    }
}


/*
 * Test cases (2-3) briefly; the real tests of Expr and TypeName
 * resolving are done in ExprScoping, et al.
 */

class Other {
    M field;

    int x = field.m();
    int y = M.m();
}
class M {
    static int m() { return 0; }
}
