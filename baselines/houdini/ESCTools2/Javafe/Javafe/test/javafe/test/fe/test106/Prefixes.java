package P;

/**
 ** Test that when computing canonical prefixes (see
 ** TypeScoping.java), we use the shortest package name (if case 1d)
 ** and the appropriate extension (see steps 2-3).
 **
 ** Note this file must be used with Prefixes1.java at the same time
 ** (both on the command line).
 **
 ** Note: This is technically an illegal program because a package and
 ** a class have the same name...  Neither we or javac actually checks
 ** this, though...
 **/


/*
 * Verify that we choose the shortest package name possible:
 */

class Q {
    static class C { static int f = 0; }   // class P.(Q.C)
}
class TestShortest {
    int x = P.Q.C.f;    // P.(Q.C) overrides (P.Q).C!
    int y = P.Q.C.g;    // error (we do not backtrack)
}


/*
 * Verify that we extend the base typename as long as we can until we
 * get to a field access or something that is not a type member access:
 */

class C {
    static int f = 0;

    static C F;

    static class F { static boolean f = false; }
}
class C2 {
    static class D {
	static C F;

	static class F { static boolean f = false; }
    }
}
class TestLongest {
    int x = C.F.f;      // field F overrides type member F...
    int y = C2.D.F.f;   // field F overrides type member F...
    int z = C.F.z;      // error: no such member z in C.F (of type P.C)
}


/*
 * Check particularly, the case where the field is inherited...
 */

class Inherited implements CF {
    static class F { static boolean f = false; }

    int x = Inherited.F.f;   // inherited field F overrides type member F ...
}
interface CF { C F = null; }


/*
 * Verify that we extend both when type member is inherited from a
 * class, and from an interface:
 */

class Extend extends withF implements withG {
    int x = Extend.F.f;   // see type members from superclasses
    int y = Extend.G.g;   // see type members from superinterfaces
}
class withF { static class F { static int f = 0; } }
interface withG { class G { static int g = 0; } }


/*
 * Verify that the lowest type member from a superclass wins when we extend
 */

class Lower extends Lower2 {
    int x = F.f;     // lower F wins...
}
class Lower2 extends Lower3 {
    static class F { static int f = 0; }
}
class Lower3 {
    static class F { static boolean f = false; }
}


/*
 * Verify that when resolving typenames, we ignore fields
 */

class TypeNames {
    int C;

    static class C {
	int F;
	static class F { }
    }

    Object test = new C.F();   // works because we ignore fields
}
