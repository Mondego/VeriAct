/**
 ** Test the scoping of Expr names (this does not include method
 ** invocations), part I.  The correct rules are:
 **
 **
 ** The algorithm for "resolving"(+) an Expr name n.N is as follows:
 **
 **
 ** (1a) n denotes the lexically-innermost binding of the given
 **      occurrence of n that is a field or local variable
 **      declaration; an ambiguous field name error results in the
 **      case of a tie.  For this purpose, a field F is considered to
 **      be located in the lexically-innermost type enclosing the name
 **      n.N that inherits or defines F.(++)
 **
 ** (1b) If no such binding of n exists, then a canonical prefix(+++), C,
 **      of n.N denotes a type.  Let C.f.N' = n.N.
 **
 ** (1c) If (1a) and (1b) do not apply, then n.N does not denote a Expr.
 **
 **
 ** (2) n.N then denotes n (if 1a) or C.f (if 1b) dereferenced by the
 **     fields .f1.f2.f3 ... .fn = N (if 1a) or N' (if 1b).  (An
 **     ambiguous field name error may result here if an attempt is
 **     made to access field f of an object or type with multiple
 **     (inherited(++)) fields named f.)
 **
 **
 ** (+) - Name resolution in the front end only does part of this
 **       algorithm.  In particular, it does not resolve fields.  That
 **       is done later by the type checking phase.
 **
 ** (++) - A field is inherited from a supertype by a subtype unless
 **        the subtype contains a field with the same name.
 **
 ** (+++) - The rules for determining the canonical typename prefix (if
 **         any) of a name may be found in TypeScoping.java.
 **/


/*
 * Verify that the innermost of two fields/locals/types wins:
 */

class SameKind1 {
    int f;
    static class T { static int f; }

    static class I1 {
	boolean f;
	static class T { static boolean f; };

	boolean g = f;          // inner field hides outer field
	boolean g2 = T.f;       // inner type hides outer type
    }
}
class SameKind2 {
    void m() {
	int l = 0;
	
	class I {
	    void n() {
		boolean l = false;     // error (local hiding a local) // Not an error in javac

		boolean s = l;         // inner local hides outer local
	    }
	}
    }
}


/*
 * Verify that locals and fields block each other:
 */

class LocalsVsFields {
    int f;

    void m() {
	boolean f = false;

	boolean g = f;           // inner local hides outer field!

	class I1 {

	    String f = "";

	    void m() {
		f = "hello";     // inner field hides outer local & field!
	    }
	}
    }
}


/*
 * Verify that locals and fields always override types, even if
 * declared outside the type declaration:
 */

class LocalsVsTypes {
    class f {};

    void m() {
	final boolean f = false;

	boolean g = f;       // inner local hides outer class!

	class I1 {

	    class f {}

	    void m() {
		if (f) ;     // inner class does not hide outer local!
	    }
	}
    }
}
class TypesVsFields {
    int f;

    class I1 {
	class f {}

	int g = f;           // inner type does not hide outer field

	class I2 {
	    boolean f;

	    boolean g = f;   // inner field hides outer type and field
	}
    }
}


/*
 * Verify that duplicate names at the same level (in the same type) is
 * an error.
 */
class DupNames1 {
    int f;
    boolean f;      // error: duplicate fields
}
class DupNames2 {
    class f{};
    class f{};      // error: duplicate types
}
class DupNames3 {
    void m() {
	int f;
	boolean f;      // error: duplicate locals
    }
}


/*
 * Verify that fields in supertypes act like they were directly
 * present in the subtypes (except for duplicate detection):
 *
 * Note: javac enforces an additional type checking rule we don't
 * which makes the lines with a [error] an error.
 */
interface intF { int f=0; }
class booleanF { boolean f; }
class ISameKind1 {
    int f;

    static class I1 extends booleanF {
	boolean g = f;      // [error] inherited inner field hides outer field
    }
}
class ILocalsVsFields {
    void m() {
	boolean f = false;

	class I1 implements intF {

	    void m() {
		int g = f; // [error] inherited inner field hides outer local!
	    }
	}
    }
}
class ITypesVsFields {
    class f {}

    class I2 extends booleanF {
	boolean g = f;         // inherited inner field hides outer type
    }
}
class IDupNames1 implements intF {
    boolean f;      // no error!
}


/*
 * The remaining cases are tested in TypeScoping.java...
 */
