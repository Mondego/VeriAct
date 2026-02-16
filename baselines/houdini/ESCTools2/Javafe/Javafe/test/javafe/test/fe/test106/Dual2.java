/**
 ** Test ambiguous references to type members (multiple parts)
 **/

class Dual2 extends SuperType implements SuperInterface {
    // Indirect reference
    int k = Dual2.T.i;            // error: ambiguous access to T

    // Direct references are ok:
    int x = SuperType.T.i;
    boolean y = SuperInterface.T.i;

}

class SuperType {
    static class T { static int i; }
}

interface SuperInterface {
    static class T { static boolean i; }
}
