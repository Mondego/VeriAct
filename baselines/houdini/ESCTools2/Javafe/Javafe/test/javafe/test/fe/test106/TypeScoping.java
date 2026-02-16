/**
 ** Test the scoping of Expr names (this does not include method
 ** invocations), part II: Exprs's which start with a typename.
 **
 **
 ** The algorithm for determining the canonical typename prefix (if
 ** any) of the name n.N and the class it denotes:
 **
 **
 ** (1a) n denotes the lexically-innermost binding of the given
 **      occurrence of n that is a type member or local type
 **      declaration; an ambiguous typename error results in the case
 **      of a tie.  For this purpose, a type member T is considered to
 **      be located in the lexically-innermost type enclosing the name
 **      n.N that inherits or defines T.(+)
 **
 ** (1b) If no such binding exists, then attempt to resolve n to an
 **      outside type defined in the same compilation unit as us.

 ** (1c) If unsuccessful, then attempt to resolve n using the imports
 **      (including the default ones for our package and java.lang.*)
 **      of our compilation unit *without* using inheritance.(++)
 **
 ** (1d) If unsuccessful, then let P.I be the smallest prefix of n.N
 **      such that there is an fully-qualified outside type with
 **      simple name I in package P (this ignores imports).  Let
 **      P.I.N' = n.N.
 **
 ** (1e) If (1a-d) do not apply, then n.N does not have a
 **      canonical typename prefix.
 **
 **
 ** (2) Let C.N' = n.N where C = n (in cases 1a-1c) or P.I (in case
 **     1d) and T be the type n denotes (in cases 1a-1c) or P.I (in
 **     case 1d).
 **
 ** (3) While .N' = .f1.N'', .f1 does not name a field of T (including
 **     fields inherited from supertypes), but does name an
 **     (inherited(+)) type member of T, do
 **
 **        set C = C.f1, N' = N'', T = C.f1(+++)
 **
 ** (4) The canonical prefix is C and it denotes T.
 **
 **
 ** (+) - A type member is inherited from a supertype by a subtype
 **       unless the subtype contains a type member with the same name.
 **
 ** (++) - Aka, each import of the form N'.* or N'.n results in a
 **        lookup of N'.n using the normal 1.1 typename lookup rules
 **        (see below) *except* that types are not considered to ever
 **        be inherited.  Single-imports have priority over
 **        demand-imports.  If two classes remain, then error due to
 **        ambiguity.
 **
 ** (+++) - If C inherits more than 1 type member named f1, then an
 **         ambiguous typename error results.
 **
 **
 ** The algorithm for resolving the typename N:
 **
 ** (1) Determine the canonical prefix C of N, pretending that no
 **     fields exist.  If it does not exist, or C != N, then fail.
 **
 ** (2) Otherwise, we resolve N to the class denoted by its canonical
 **     prefix.
 **
 ** Exception: when resolving typenames in imports, we do not consider
 **            type members to be inherited from classes.
 **/


/*
 * Verify types can be inherited both from classes and from interfaces:
 *
 * We test the ambiguous cases of (1a) and (3) in Dual*.java.
 */

class Inheritance extends intT implements booleanF {
    int x = T.f;           // types inherited from classes
    int y = U.f;           // types inherited from interfaces
}
class intT {
    static class T { static int f = 0; }
}
interface booleanF {
    class U { static int f = 0; }
}


/*
 * Verify that fields in superclasses act like they were directly
 * present in the subtypes (except for duplicate detection):
 *
 * Note: javac enforces an additional type checking rule we don't
 * which makes the lines with a [error] an error.
 */

class Blocking {
    static class T { static boolean f = false; }

    static class I extends intT {
	int x = T.f;      // [error] inner inherited type blocks outer type
    }
}
class Hiding extends intT {
    static class T { static boolean f = false; }

    boolean x = T.f;  // type in subclass hides same type in superclasses
}

    
/*
 * Verify that inside definitions have priority over imports and the like:
 */

class Outside {}
class Priorty {
    static class Outside { static int f = 0; }

    int x = Outside.f;    // inner type blocks compilation-level type
}


/*
 * Verify that local class definitions work properly:
 */
class LocalDefs {
    void m() {
	class T {};

	T x = new T();    // no error: T is in scope here
    }
}


/**
 ** Most of rules (1b-c) are tested by previous Java 1.0 regression
 ** tests; in this test suite we test only the differences (1c
 ** re. inheritence) in Imports.java.
 **
 ** We test (1d-3) in Prefixes.java.
 **/
