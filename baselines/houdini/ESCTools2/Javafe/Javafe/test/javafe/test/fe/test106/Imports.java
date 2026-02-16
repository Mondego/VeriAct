package P;

import P.A.*;
import P.B.D;


/**
 ** Test that imports ignore inheritance and that we can import nested classes
 **/

class A extends B {
}
class B {
    static class C { static int f = 0; }
    static class D { static boolean f = false; }
}


class Test {
    int x = C.f;             // error: can't access C without inheritance
    int y = P.A.C.f;         // ok w/o use of imports, though
    boolean z = D.f;         // ok if no inheritance required
}
